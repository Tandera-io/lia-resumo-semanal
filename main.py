from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import unicodedata
import logging
import os
import uuid
from datetime import datetime, timedelta
import json
from supabase import create_client, Client
import anthropic
from apscheduler.schedulers.background import BackgroundScheduler
from zoneinfo import ZoneInfo

app = FastAPI(
    title="LIA Resumo Semanal API",
    description="API para geração de resumos semanais executivos usando Anthropic Claude",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adicionar middleware de tenant (DEPOIS do CORS para que OPTIONS seja processado primeiro)
from middleware.tenant import TenantMiddleware, get_tenant_context
app.add_middleware(TenantMiddleware)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL")

logger = logging.getLogger("lia_resumo_semanal")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = True

def _clean_env_value(v: str) -> str:
    return (v or "").strip().strip('"').strip("'")

def get_supabase_client() -> Client:
    """Cria cliente Supabase com suporte multi-tenancy"""
    # Tentar obter credenciais do contexto do tenant (multi-tenancy)
    try:
        tenant_ctx = get_tenant_context()
        if tenant_ctx.tenant_slug and tenant_ctx.tenant_data:
            url = tenant_ctx.get_supabase_url()
            key = tenant_ctx.get_anon_key()
            
            if url and key:
                logger.info(f"[Supabase] Usando credenciais do tenant: {tenant_ctx.tenant_slug}")
                return create_client(url, key)
    except Exception as e:
        logger.debug(f"[Supabase] Tenant context não disponível, usando credenciais padrão: {e}")
    
    # Fallback para credenciais padrão do .env
    SUPABASE_URL = _clean_env_value(os.getenv("SUPABASE_URL") or "")
    SUPABASE_KEY = _clean_env_value(os.getenv("SUPABASE_KEY") or "")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_admin() -> Client:
    """Cliente Supabase com privilégios administrativos (bypass RLS) - suporte multi-tenancy"""
    # Tentar obter credenciais do contexto do tenant (multi-tenancy)
    try:
        tenant_ctx = get_tenant_context()
        if tenant_ctx.tenant_slug and tenant_ctx.tenant_data:
            url = tenant_ctx.get_supabase_url()
            service_key = tenant_ctx.get_service_key()
            
            if url and service_key:
                logger.info(f"[Supabase] Usando SERVICE_ROLE do tenant: {tenant_ctx.tenant_slug}")
                return create_client(url, service_key)
    except Exception as e:
        logger.debug(f"[Supabase] Tenant context SERVICE_ROLE não disponível, usando credenciais padrão: {e}")
    
    # Fallback para credenciais padrão do .env
    SUPABASE_URL = _clean_env_value(os.getenv("SUPABASE_URL") or "")
    service_key = _clean_env_value(os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY") or "")
    return create_client(SUPABASE_URL, service_key)

class WeeklySummaryResponse(BaseModel):
    id: str
    project_id: str
    year: int
    iso_week: int
    period_start: str
    period_end: str
    title: str
    executive_summary: str
    kpis: Optional[Any] = None
    risks: Optional[Any] = None
    next_actions: Optional[Any] = None
    meetings_count: int
    tasks_count: int
    created_at: str

class WeeklySummaryListResponse(BaseModel):
    summaries: List[WeeklySummaryResponse]
    total: int

def _prev_week_range():
    today = datetime.now()
    days_since_monday = today.weekday()
    last_monday = today - timedelta(days=days_since_monday + 7)
    last_sunday = last_monday + timedelta(days=6)
    year, iso_week, _ = last_monday.isocalendar()
    return year, iso_week, last_monday, last_sunday

def _fmt_br(dt):
    return dt.strftime('%d/%m/%Y') if dt else ''

def _validate_uuid_string(value: str) -> str:
    try:
        uuid.UUID(value)
        return value
    except Exception:
        raise HTTPException(status_code=422, detail="project_id must be a valid UUID")

def _extract_data(resp):
    """Safely extract .data from Supabase responses without assuming truthiness.

    Some responses (like empty lists) are falsy; avoid falling back to dict.get
    on non-dict objects which raises attribute errors.
    """
    if resp is None:
        return []
    data_attr = getattr(resp, 'data', None)
    if data_attr is not None:
        return data_attr
    if isinstance(resp, dict):
        return resp.get('data') or []
    return []

def _extract_count(resp) -> int:
    if resp is None:
        return 0
    cnt = getattr(resp, 'count', None)
    if isinstance(cnt, int):
        return cnt
    if isinstance(resp, dict):
        c = resp.get('count')
        if isinstance(c, int):
            return c
    return 0

def _anthropic_json(prompt: str) -> Dict[str, Any]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    candidate_models = [m for m in [
        ANTHROPIC_MODEL,
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-5-sonnet-20240620",
    ] if m]
    last_err = None
    for model_name in candidate_models:
        try:
            msg = client.messages.create(
                model=model_name,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = msg.content[0].text
            return json.loads(_extract_json_text(response_text))
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("No Anthropic model available for messages API")

def _extract_json_text(text: str) -> str:
    try:
        # Quick path if already a clean JSON string
        json.loads(text)
        return text
    except Exception:
        import re
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            raise ValueError("No valid JSON found in provider response")
        return m.group(0)


def _log_context(label: str, payload: Dict[str, Any]) -> None:
    try:
        logger.info("%s %s", label, json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        logger.info("%s %s", label, payload)


def _anthropic_text(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    candidate_models = [m for m in [
        ANTHROPIC_MODEL,
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-5-sonnet-20240620",
    ] if m]
    last_err = None
    for model_name in candidate_models:
        try:
            msg = client.messages.create(
                model=model_name,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("No Anthropic model available for plain text messages")

SUMMARY_TASK_STATUS_QUERY = [
    "A fazer", "A Fazer", "a fazer",
    "Em Progresso", "Em progresso", "em progresso",
    "Em Andamento", "Em andamento", "em andamento",
    "Concluidas", "Concluídas", "Concluida", "Concluída",
    "concluidas", "Concluído", "Concluido", "concluido", "Concluídos", "concluidos"
]

SUMMARY_TASK_STATUS_ALLOW = {
    "a fazer",
    "em progresso",
    "em andamento",
    "concluida",
    "concluidas",
    "concluido",
    "concluidos"
}


def _normalize_status(value: Optional[str]) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize('NFKD', value)
    normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.strip().lower()


def _should_include_status(value: Optional[str]) -> bool:
    return _normalize_status(value) in SUMMARY_TASK_STATUS_ALLOW


def _generate_claude_summary(project_name: str, meetings: List[Dict], tasks: List[Dict], period_label: str) -> Dict[str, Any]:
    meetings_md = "\n".join([
        f"- {m['date']} • {m['title']}" for m in meetings
    ]) if meetings else "- (nenhuma)"
    tasks_md = "\n".join([
        f"- {t['status'].title()} • {t['title']} — prazo: {t['deadline']}"
        for t in tasks
    ]) if tasks else "- (nenhuma)"

    prompt = f"""Você é um analista executivo sênior responsável por relatar a última semana do projeto a uma diretoria exigente.

Contexto do projeto: {project_name}
Período analisado: {period_label}

Reuniões realizadas na semana anterior:
{meetings_md}

Tarefas do projeto em status ativos ("A fazer", "Em Progresso", "Concluídas"):
{tasks_md}

Produza apenas o resumo executivo completo da semana, em português brasileiro, seguindo estas regras:
- Escreva 4 a 6 parágrafos em prosa contínua (sem tópicos, sem listas, sem JSON).
- Cite explicitamente fatos concretos vindos das reuniões e tarefas (datas, prazos, responsáveis, status, decisões, riscos percebidos, impactos em entregas).
- Explique como cada evento afetou o andamento do projeto e quais ações estão em curso.
- Se algum dado estiver ausente, declare isso de forma transparente ao invés de inventar.
- Mantenha tom executivo, claro e objetivo, adequado a C-level.
- Finalize com um parágrafo sintetizando próximos passos imediatos já evidentes nas fontes acima.
- Responda SOMENTE com o texto final do resumo executivo.
"""

    try:
        summary_text = _anthropic_text(prompt)
        if not summary_text:
            raise ValueError("Modelo retornou texto vazio")
        return {
            "executive_summary": summary_text,
            "content": summary_text
        }

    except Exception as e:
        fallback_exec = (
            f"Resumo semanal do projeto {project_name} ({period_label}). "
            f"Registrei {len(meetings)} reunião(ões) e {len(tasks)} tarefa(s) em andamento. "
            f"Não consegui elaborar uma análise detalhada devido a um erro técnico: {str(e)}"
        )
        return {
            "executive_summary": fallback_exec,
            "content": fallback_exec
        }

def _generate_claude_week_plan(project_name: str, tasks: List[Dict], period_label: str) -> Dict[str, Any]:
    tasks_md = "\n".join([f"- {t['deadline']} • {t['title']} — status: {t['status']}" for t in tasks]) if tasks else "- (nenhuma)"
    prompt = f"""Você é um PMO experiente. Monte um PLANO DA SEMANA para o projeto abaixo.

Projeto: {project_name}
Período (semana atual): {period_label}

Insumos:
Tarefas com prazo nesta semana:
{tasks_md}

Instruções:
- Responda APENAS um JSON válido, sem texto fora do JSON.
- Campos obrigatórios do JSON:
  - "executive_summary": 1-3 parágrafos com foco da semana, prioridades, dependências críticas e decisões esperadas.
  - "kpis": Métricas-alvo da semana (ex: {{"tasks_due": "X", "at_risk": "Y", "throughput_target": "N tarefas"}})
  - "kpi_explanations": Objeto com justificativas curtas por KPI-alvo (ex: {{"at_risk": "tarefas com dependência externa sem confirmação"}})
  - "risks": Array de riscos desta semana com {{"description", "impact", "likelihood", "trigger_signals", "mitigation", "owner"}}
  - "next_actions": Array com {{"action", "responsible", "deadline", "priority", "rationale"}} priorizadas (alto → baixo)
  - "agenda_topics": Tópicos recomendados para as reuniões ({{"topic", "reason"}})

- Foque em pendências, riscos de prazo, dependências e decisões necessárias.
- Escreva em português brasileiro."""
    try:
        data = _anthropic_json(prompt)
        if "agenda_topics" not in data:
            data["agenda_topics"] = []
        return data
    except Exception as e:
        return {
            "executive_summary": f"Plano da semana para {project_name} ({period_label}). Análise detalhada indisponível: {str(e)}",
            "kpis": {},
            "risks": [],
            "next_actions": [],
            "agenda_topics": []
        }

def _current_week_range():
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    year, iso_week, _ = monday.isocalendar()
    return year, iso_week, monday, sunday

def _next_version_for_week(supabase: Client, project_id: str, year: int, iso_week: int) -> int:
    existing = (supabase
                .table('project_weekly_reports')
                .select('version')
                .eq('project_id', project_id)
                .eq('year', int(year))
                .eq('iso_week', int(iso_week))
                .order('version', desc=True)
                .limit(1)
                .execute())
    rows = _extract_data(existing)
    if rows:
        try:
            return int(rows[0].get('version') or 0) + 1
        except Exception:
            return 1
    return 1

def _insert_report_with_retry(supabase_admin: Client, payload: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
    """Insert report, bumping version if unique constraint is hit.

    This protects against races or read inconsistencies by retrying with
    version+1 when Postgres returns constraint 23505.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            resp = supabase_admin.table('project_weekly_reports').insert(payload).execute()
            data = _extract_data(resp)
            if not data:
                raise RuntimeError("Insert returned empty data")
            return data[0]
        except Exception as e:
            s = str(e)
            if '23505' in s or 'duplicate key' in s:
                # bump version and retry
                payload['version'] = int(payload.get('version', 1)) + 1
                attempt += 1
                continue
            raise
    raise RuntimeError("Could not insert report after retrying versions")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "lia-resumo-semanal"}

@app.post("/api/projects/{project_id}/weekly-summary", response_model=WeeklySummaryResponse)
async def generate_weekly_summary(project_id: str):
    try:
        project_id = _validate_uuid_string(project_id)
        supabase = get_supabase_client()
        year, iso_week, start, end = _prev_week_range()

        proj = supabase.table('projects').select('id,name').eq('id', project_id).single().execute()
        proj_data = _extract_data(proj) or {}
        if isinstance(proj_data, list):
            proj_data = proj_data[0] if proj_data else {}
        if not isinstance(proj_data, dict):
            proj_data = {}
        project_name = proj_data.get('name') or 'Projeto'

        _log_context("weekly_summary.project", {
            "project_id": project_id,
            "project_name": project_name,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
        })

        trans = (supabase
                 .table('transcriptions')
                 .select('id,title,reuniao,created_at')
                 .eq('project_id', project_id)
                 .gte('created_at', start.isoformat())
                 .lte('created_at', end.isoformat())
                 .execute())
        trans_rows = _extract_data(trans)
        meetings = []
        for r in trans_rows:
            created_at = r.get('created_at')
            try:
                created_dt = datetime.fromisoformat((created_at or '').replace('Z', '+00:00'))
            except Exception:
                created_dt = start
            meetings.append({
                'id': r.get('id'), 
                'title': r.get('reuniao') or r.get('title') or 'Reunião', 
                'date': _fmt_br(created_dt)
            })

        _log_context("weekly_summary.meetings", {
            "project_id": project_id,
            "count": len(meetings),
            "items": meetings
        })

        tasks_q = (supabase
                   .table('kanban_tasks')
                   .select('id,title,status,deadline,description,updated_at')
                   .eq('project_id', project_id)
                   .in_('status', SUMMARY_TASK_STATUS_QUERY)
                   .execute())
        task_rows = _extract_data(tasks_q)
        tasks = []
        for t in task_rows:
            status_value = t.get('status')
            if not _should_include_status(status_value):
                continue
            ddl = t.get('deadline')
            try:
                ddl_dt = datetime.fromisoformat((ddl or '').replace('Z', '+00:00')) if ddl else None
            except Exception:
                ddl_dt = None
            deadline_display = _fmt_br(ddl_dt) if ddl_dt else 'Sem prazo definido'
            note = (t.get('description') or '').strip()
            tasks.append({
                'id': t.get('id'),
                'title': t.get('title') or 'Tarefa',
                'status': status_value or 'Indefinido',
                'deadline': deadline_display,
                'notes': note
            })
        tasks.sort(key=lambda item: (_normalize_status(item.get('status')), item.get('deadline') or '', item.get('title') or ''))

        _log_context("weekly_summary.tasks", {
            "project_id": project_id,
            "count": len(tasks),
            "items": tasks
        })

        period_label = f"{_fmt_br(start)} a {_fmt_br(end)}"
        
        claude_result = _generate_claude_summary(project_name, meetings, tasks, period_label)
        _log_context("weekly_summary.summary_generated", {
            "project_id": project_id,
            "summary_preview": claude_result.get('executive_summary', '')[:500]
        })
        
        existing = (supabase.table('project_weekly_reports')
                    .select('id, version')
                    .eq('project_id', project_id)
                    .eq('year', int(year))
                    .eq('iso_week', int(iso_week))
                    .order('version', desc=True)
                    .limit(1)
                    .execute())
        rows = _extract_data(existing)
        version = 1
        if rows:
            version = int(rows[0].get('version') or 1) + 1

        title = f"Resumo Semanal — Semana {iso_week}, {year}"
        
        supabase_admin = get_supabase_admin()
        insert = (supabase_admin.table('project_weekly_reports').insert({
            'project_id': project_id,
            'year': int(year),
            'iso_week': int(iso_week),
            'period_start': start.isoformat(),
            'period_end': end.isoformat(),
            'version': version,
            'title': title,
            'content_markdown': claude_result.get('content'),
            'kpis': None,
            'risks': None,
            'meta': {
                'generated_from': 'lia-resumo-semanal',
                'report_type': 'weekly_summary',
                'meetings': len(meetings),
                'tasks': len(tasks),
                'context': {
                    'meetings': meetings,
                    'tasks': tasks
                },
                'executive_summary_raw': claude_result.get('executive_summary')
            }
        }).execute())
        
        data = _extract_data(insert)
        report_id = data[0]['id']
        
        return WeeklySummaryResponse(
            id=report_id,
            project_id=project_id,
            year=int(year),
            iso_week=int(iso_week),
            period_start=start.isoformat(),
            period_end=end.isoformat(),
            title=title,
            executive_summary=claude_result.get('executive_summary', ''),
            kpis=None,
            risks=None,
            next_actions=None,
            meetings_count=len(meetings),
            tasks_count=len(tasks),
            created_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}/weekly-summaries", response_model=WeeklySummaryListResponse)
async def list_weekly_summaries(project_id: str, limit: int = 10, offset: int = 0):
    try:
        project_id = _validate_uuid_string(project_id)
        supabase = get_supabase_client()
        
        query = (supabase
                .table('project_weekly_reports')
                .select('*')
                .eq('project_id', project_id)
                .order('year', desc=True)
                .order('iso_week', desc=True)
                .range(offset, offset + limit - 1))
        
        result = query.execute()
        rows = _extract_data(result)
        if not isinstance(rows, list):
            rows = []
        
        summaries = []
        for row in rows:
            meta = row.get('meta', {}) if isinstance(row, dict) else {}
            stored_kpis = row.get('kpis') if isinstance(row, dict) else None
            stored_risks = row.get('risks') if isinstance(row, dict) else None
            summary_text = ''
            if isinstance(meta, dict):
                summary_text = meta.get('executive_summary_raw') or ''
            if not summary_text:
                summary_text = row.get('content_markdown', '') if isinstance(row, dict) else ''
            summaries.append(WeeklySummaryResponse(
                id=row['id'],
                project_id=row['project_id'],
                year=row['year'],
                iso_week=row['iso_week'],
                period_start=row['period_start'],
                period_end=row['period_end'],
                title=row['title'],
                executive_summary=summary_text,
                kpis=stored_kpis,
                risks=stored_risks,
                next_actions=meta.get('next_actions') if isinstance(meta, dict) else None,
                meetings_count=meta.get('meetings', 0),
                tasks_count=meta.get('tasks', 0),
                created_at=row['created_at']
            ))
        
        count_result = (supabase
                       .table('project_weekly_reports')
                       .select('id', count='exact')
                       .eq('project_id', project_id)
                       .execute())
        total = _extract_count(count_result)
        if total == 0:
            total = len(summaries)
        
        return WeeklySummaryListResponse(summaries=summaries, total=total)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/{project_id}/weekly-plan", response_model=WeeklySummaryResponse)
async def generate_weekly_plan(project_id: str):
    try:
        project_id = _validate_uuid_string(project_id)
        supabase = get_supabase_client()
        year, iso_week, start, end = _current_week_range()

        proj = supabase.table('projects').select('id,name').eq('id', project_id).single().execute()
        proj_data = _extract_data(proj) or {}
        if isinstance(proj_data, list):
            proj_data = proj_data[0] if proj_data else {}
        project_name = proj_data.get('name') or 'Projeto'

        _log_context("weekly_plan.project", {
            "project_id": project_id,
            "project_name": project_name,
            "period_start": start.isoformat(),
            "period_end": end.isoformat()
        })

        tasks_q = (supabase
                   .table('kanban_tasks')
                   .select('id,title,status,deadline')
                   .eq('project_id', project_id)
                   .gte('deadline', start.isoformat())
                   .lte('deadline', end.isoformat())
                   .execute())
        task_rows = _extract_data(tasks_q)
        tasks = []
        for t in task_rows:
            ddl = t.get('deadline')
            try:
                ddl_dt = datetime.fromisoformat((ddl or '').replace('Z', '+00:00')) if ddl else None
            except Exception:
                ddl_dt = None
            tasks.append({
                'id': t.get('id'),
                'title': t.get('title') or 'Tarefa',
                'status': (t.get('status') or '').lower(),
                'deadline': _fmt_br(ddl_dt) if ddl_dt else '—'
            })

        _log_context("weekly_plan.tasks", {
            "project_id": project_id,
            "count": len(tasks),
            "items": tasks
        })

        period_label = f"{_fmt_br(start)} a {_fmt_br(end)}"
        plan = _generate_claude_week_plan(project_name, tasks, period_label)
        _log_context("weekly_plan.generated", {
            "project_id": project_id,
            "summary_preview": plan.get('executive_summary', '')[:500]
        })

        title = f"Plano da Semana — Semana {iso_week}, {year}"

        supabase_admin = get_supabase_admin()
        version = _next_version_for_week(supabase_admin, project_id, year, iso_week)
        payload = {
            'project_id': project_id,
            'year': int(year),
            'iso_week': int(iso_week),
            'period_start': start.isoformat(),
            'period_end': end.isoformat(),
            'version': version,
            'title': title,
            'content_markdown': plan.get('executive_summary', ''),
            'kpis': plan.get('kpis', {}),
            'risks': plan.get('risks', []),
            'meta': {
                'generated_from': 'lia-resumo-semanal',
                'report_type': 'weekly_plan',
                'agenda_topics': plan.get('agenda_topics', []),
                'tasks': len(tasks),
                'next_actions': plan.get('next_actions', [])
            }
        }
        inserted = _insert_report_with_retry(supabase_admin, payload)
        report_id = inserted['id']

        return WeeklySummaryResponse(
            id=report_id,
            project_id=project_id,
            year=int(year),
            iso_week=int(iso_week),
            period_start=start.isoformat(),
            period_end=end.isoformat(),
            title=title,
            executive_summary=plan.get('executive_summary', ''),
            kpis=plan.get('kpis', {}),
            risks=plan.get('risks', []),
            next_actions=plan.get('next_actions', []),
            meetings_count=0,
            tasks_count=len(tasks),
            created_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _friday_summary_job():
    try:
        tz = ZoneInfo("America/Sao_Paulo")
        now = datetime.now(tz)
        year, iso_week, start, _ = _current_week_range()
        end = now.replace(tzinfo=None)
        supabase = get_supabase_client()
        projects = _extract_data(supabase.table('projects').select('id,name').execute())
        for p in projects:
            pid = p.get('id')
            try:
                generate_summary_for_range(pid, start, end)
            except Exception:
                continue
    except Exception:
        pass

def generate_summary_for_range(project_id: str, start: datetime, end: datetime):
    # Internal helper used by Friday job
    supabase = get_supabase_client()
    proj = supabase.table('projects').select('id,name').eq('id', project_id).single().execute()
    proj_data = _extract_data(proj) or {}
    if isinstance(proj_data, list):
        proj_data = proj_data[0] if proj_data else {}
    project_name = proj_data.get('name') or 'Projeto'

    _log_context("scheduled_summary.project", {
        "project_id": project_id,
        "project_name": project_name,
        "period_start": start.isoformat(),
        "period_end": end.isoformat()
    })

    trans = (supabase
             .table('transcriptions')
             .select('id,title,reuniao,created_at')
             .eq('project_id', project_id)
             .gte('created_at', start.isoformat())
             .lte('created_at', end.isoformat())
             .execute())
    trans_rows = _extract_data(trans)
    meetings = []
    for r in trans_rows:
        created_at = r.get('created_at')
        try:
            created_dt = datetime.fromisoformat((created_at or '').replace('Z', '+00:00'))
        except Exception:
            created_dt = start
        meetings.append({'id': r.get('id'), 'title': r.get('reuniao') or r.get('title') or 'Reunião', 'date': _fmt_br(created_dt)})

    _log_context("scheduled_summary.meetings", {
        "project_id": project_id,
        "count": len(meetings),
        "items": meetings
    })

    tasks_q = (supabase
               .table('kanban_tasks')
               .select('id,title,status,deadline,description,updated_at')
               .eq('project_id', project_id)
               .in_('status', SUMMARY_TASK_STATUS_QUERY)
               .execute())
    task_rows = _extract_data(tasks_q)
    tasks = []
    for t in task_rows:
        status_value = t.get('status')
        if not _should_include_status(status_value):
            continue
        ddl = t.get('deadline')
        try:
            ddl_dt = datetime.fromisoformat((ddl or '').replace('Z', '+00:00')) if ddl else None
        except Exception:
            ddl_dt = None
        deadline_display = _fmt_br(ddl_dt) if ddl_dt else 'Sem prazo definido'
        note = (t.get('description') or '').strip()
        tasks.append({
            'id': t.get('id'),
            'title': t.get('title') or 'Tarefa',
            'status': status_value or 'Indefinido',
            'deadline': deadline_display,
            'notes': note
        })
    tasks.sort(key=lambda item: (_normalize_status(item.get('status')), item.get('deadline') or '', item.get('title') or ''))

    _log_context("scheduled_summary.tasks", {
        "project_id": project_id,
        "count": len(tasks),
        "items": tasks
    })

    period_label = f"{_fmt_br(start)} a {_fmt_br(end)}"
    result = _generate_claude_summary(project_name, meetings, tasks, period_label)
    _log_context("scheduled_summary.summary_generated", {
        "project_id": project_id,
        "summary_preview": result.get('executive_summary', '')[:500]
    })

    year, iso_week, _, _ = _current_week_range()
    title = f"Resumo Semanal — Semana {iso_week}, {year} (Parcial)"
    supabase_admin = get_supabase_admin()
    version = _next_version_for_week(supabase_admin, project_id, year, iso_week)
    payload = {
        'project_id': project_id,
        'year': int(year),
        'iso_week': int(iso_week),
        'period_start': start.isoformat(),
        'period_end': end.isoformat(),
        'version': version,
        'title': title,
        'content_markdown': result.get('content'),
        'kpis': None,
        'risks': None,
        'meta': {
            'generated_from': 'lia-resumo-semanal',
            'report_type': 'friday_partial_summary',
            'meetings': len(meetings),
            'tasks': len(tasks),
            'context': {
                'meetings': meetings,
                'tasks': tasks
            },
            'executive_summary_raw': result.get('executive_summary')
        }
    }
    _insert_report_with_retry(supabase_admin, payload)

@app.on_event("startup")
def _startup_jobs():
    try:
        tz = ZoneInfo("America/Sao_Paulo")
        scheduler = BackgroundScheduler(timezone=tz)
        # Sexta às 17:00 — resumo parcial da semana
        scheduler.add_job(_friday_summary_job, 'cron', day_of_week='fri', hour=17, minute=0, id='friday_summary')
        # Segunda às 07:00 — plano da semana (para todos os projetos)
        scheduler.add_job(_monday_plan_job, 'cron', day_of_week='mon', hour=7, minute=0, id='monday_plan')
        scheduler.start()
        app.state.scheduler = scheduler
    except Exception:
        pass

@app.on_event("shutdown")
def _shutdown_jobs():
    sched = getattr(app.state, 'scheduler', None)
    if sched:
        try:
            sched.shutdown()
        except Exception:
            pass

def _monday_plan_job():
    try:
        year, iso_week, start, end = _current_week_range()
        supabase = get_supabase_client()
        projects = _extract_data(supabase.table('projects').select('id,name').execute())
        for p in projects:
            pid = p.get('id')
            try:
                # Use API logic to create a plan per project
                # Reuse generate_weekly_plan core logic
                # Fetch tasks and call AI, then insert (same as endpoint)
                proj = supabase.table('projects').select('id,name').eq('id', pid).single().execute()
                proj_data = _extract_data(proj) or {}
                if isinstance(proj_data, list):
                    proj_data = proj_data[0] if proj_data else {}
                project_name = proj_data.get('name') or 'Projeto'

                tasks_q = (supabase
                           .table('kanban_tasks')
                           .select('id,title,status,deadline')
                           .eq('project_id', pid)
                           .gte('deadline', start.isoformat())
                           .lte('deadline', end.isoformat())
                           .execute())
                task_rows = _extract_data(tasks_q)
                tasks = []
                for t in task_rows:
                    ddl = t.get('deadline')
                    try:
                        ddl_dt = datetime.fromisoformat((ddl or '').replace('Z', '+00:00')) if ddl else None
                    except Exception:
                        ddl_dt = None
                    tasks.append({'id': t.get('id'), 'title': t.get('title') or 'Tarefa', 'status': (t.get('status') or '').lower(), 'deadline': _fmt_br(ddl_dt) if ddl_dt else '—'})

                period_label = f"{_fmt_br(start)} a {_fmt_br(end)}"
                plan = _generate_claude_week_plan(project_name, tasks, period_label)
                title = f"Plano da Semana — Semana {iso_week}, {year}"

                supabase_admin = get_supabase_admin()
                version = _next_version_for_week(supabase_admin, pid, year, iso_week)
                payload = {
                    'project_id': pid,
                    'year': int(year),
                    'iso_week': int(iso_week),
                    'period_start': start.isoformat(),
                    'period_end': end.isoformat(),
                    'version': version,
                    'title': title,
                    'content_markdown': plan.get('executive_summary', ''),
                    'kpis': plan.get('kpis', {}),
                    'risks': plan.get('risks', []),
                    'meta': {
                        'generated_from': 'lia-resumo-semanal',
                        'report_type': 'weekly_plan',
                        'agenda_topics': plan.get('agenda_topics', []),
                        'tasks': len(tasks),
                        'next_actions': plan.get('next_actions', [])
                    }
                }
                _insert_report_with_retry(supabase_admin, payload)
            except Exception:
                continue
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
