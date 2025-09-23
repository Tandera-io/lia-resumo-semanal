from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_admin() -> Client:
    service_key = os.getenv("SUPABASE_SERVICE_KEY") or SUPABASE_KEY
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
    kpis: Dict[str, Any]
    risks: List[Dict[str, Any]]
    next_actions: List[Dict[str, Any]]
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
    """Try to get a JSON response from Anthropic using available APIs.

    Prefers completions API for broader compatibility with the pinned SDK,
    and falls back to parsing the best-effort JSON.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    last_error = None

    # Prefer messages API if available
    if hasattr(client, 'messages'):
        try:
            msg = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = msg.content[0].text
            return json.loads(_extract_json_text(response_text))
        except Exception as e:
            last_error = e

    # Fallback to completions API if available
    if hasattr(client, 'completions'):
        try:
            content = f"{getattr(anthropic, 'HUMAN_PROMPT', '\n\nHuman: ')}{prompt}{getattr(anthropic, 'AI_PROMPT', '\n\nAssistant: ')}"
            comp = client.completions.create(
                model="claude-2.1",
                max_tokens_to_sample=2000,
                temperature=0.3,
                prompt=content,
            )
            response_text = getattr(comp, 'completion', '') or str(comp)
            return json.loads(_extract_json_text(response_text))
        except Exception as e:
            last_error = e

    if last_error:
        raise last_error
    raise RuntimeError("No compatible Anthropic API (messages or completions) available")

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

def _generate_claude_summary(project_name: str, meetings: List[Dict], tasks: List[Dict], period_label: str) -> Dict[str, Any]:
    meetings_md = "\n".join([f"- {m['date']} • {m['title']}" for m in meetings]) if meetings else "- (nenhuma)"
    tasks_md = "\n".join([f"- {t['deadline']} • {t['title']} — status: {t['status']}" for t in tasks]) if tasks else "- (nenhuma)"
    
    prompt = f"""Você é um analista executivo sênior. Gere um RELATÓRIO SEMANAL executivo e detalhado para o projeto abaixo.

Projeto: {project_name}
Período: {period_label}

Insumos:
Reuniões realizadas na semana anterior:
{meetings_md}

Tarefas com prazo na semana anterior:
{tasks_md}

Instruções:
- Gere um JSON estruturado com os seguintes campos:
  - "executive_summary": Resumo executivo em 2-3 parágrafos, linguagem executiva e clara
  - "kpis": Objeto com métricas concretas (ex: {{"total_meetings": {len(meetings)}, "total_tasks": {len(tasks)}, "completion_rate": "X%", "productivity_score": "X/10"}})
  - "risks": Array de objetos com {{"description": "descrição do risco", "impact": "alto/médio/baixo", "mitigation": "ação de mitigação"}}
  - "next_actions": Array de objetos com {{"action": "ação sugerida", "responsible": "responsável sugerido", "deadline": "prazo sugerido", "priority": "alta/média/baixa"}}

- Se não houver dados suficientes, seja criativo mas realista nas sugestões
- Escreva em português brasileiro
- Responda APENAS o JSON válido, sem explicações adicionais
- Foque em insights executivos e direcionamentos estratégicos"""

    try:
        return _anthropic_json(prompt)
                
    except Exception as e:
        return {
            "executive_summary": f"Resumo semanal do projeto {project_name} ({period_label}). Principais pontos: {len(meetings)} reunião(ões) realizadas; {len(tasks)} tarefa(s) com prazo na semana. Análise detalhada indisponível devido a erro técnico: {str(e)}",
            "kpis": {
                "total_meetings": len(meetings),
                "total_tasks": len(tasks),
                "completion_rate": "N/A",
                "productivity_score": "N/A"
            },
            "risks": [{"description": "Análise de riscos indisponível", "impact": "baixo", "mitigation": "Verificar sistema de análise"}],
            "next_actions": [{"action": "Revisar dados da semana", "responsible": "Equipe", "deadline": "Próxima semana", "priority": "média"}]
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
- Responder em JSON válido com os campos:
  - "executive_summary": Uma visão concisa do foco da semana (1-2 parágrafos)
  - "kpis": Métricas-alvo para a semana (ex: {{"tasks_due": "X", "at_risk": "Y"}})
  - "risks": Array de objetos com {{"description", "impact", "mitigation"}} focados na semana
  - "next_actions": Array de objetos {{"action", "responsible", "deadline", "priority"}} sendo o plano sugerido da semana
  - "agenda_topics": Array de tópicos recomendados para reuniões da semana

- Foque em pendências, riscos de prazo, dependências e decisões necessárias
- Escreva em português brasileiro
- Responda APENAS o JSON válido, sem explicações adicionais"""
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

        period_label = f"{_fmt_br(start)} a {_fmt_br(end)}"
        
        claude_result = _generate_claude_summary(project_name, meetings, tasks, period_label)
        
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
            'content_markdown': claude_result.get('executive_summary', ''),
            'kpis': claude_result.get('kpis', {}),
            'risks': claude_result.get('risks', []),
            'meta': {
                'generated_from': 'lia-resumo-semanal',
                'report_type': 'weekly_summary',
                'meetings': len(meetings),
                'tasks': len(tasks),
                'next_actions': claude_result.get('next_actions', [])
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
            kpis=claude_result.get('kpis', {}),
            risks=claude_result.get('risks', []),
            next_actions=claude_result.get('next_actions', []),
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
            summaries.append(WeeklySummaryResponse(
                id=row['id'],
                project_id=row['project_id'],
                year=row['year'],
                iso_week=row['iso_week'],
                period_start=row['period_start'],
                period_end=row['period_end'],
                title=row['title'],
                executive_summary=row.get('content_markdown', ''),
                kpis=row.get('kpis', {}),
                risks=row.get('risks', []),
                next_actions=meta.get('next_actions', []),
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

        period_label = f"{_fmt_br(start)} a {_fmt_br(end)}"
        plan = _generate_claude_week_plan(project_name, tasks, period_label)

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
        tasks.append({'id': t.get('id'), 'title': t.get('title') or 'Tarefa', 'status': (t.get('status') or '').lower(), 'deadline': _fmt_br(ddl_dt) if ddl_dt else '—'})

    period_label = f"{_fmt_br(start)} a {_fmt_br(end)}"
    result = _generate_claude_summary(project_name, meetings, tasks, period_label)

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
        'content_markdown': result.get('executive_summary', ''),
        'kpis': result.get('kpis', {}),
        'risks': result.get('risks', []),
        'meta': {
            'generated_from': 'lia-resumo-semanal',
            'report_type': 'friday_partial_summary',
            'meetings': len(meetings),
            'tasks': len(tasks),
            'next_actions': result.get('next_actions', [])
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
