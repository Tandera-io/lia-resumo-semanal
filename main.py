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

def _generate_claude_summary(project_name: str, meetings: List[Dict], tasks: List[Dict], period_label: str) -> Dict[str, Any]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
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
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                raise ValueError("No valid JSON found in Claude response")
                
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "lia-resumo-semanal"}

@app.post("/api/projects/{project_id}/weekly-summary", response_model=WeeklySummaryResponse)
async def generate_weekly_summary(project_id: str):
    try:
        supabase = get_supabase_client()
        year, iso_week, start, end = _prev_week_range()

        proj = supabase.table('projects').select('id,name').eq('id', project_id).single().execute()
        proj_data = getattr(proj, 'data', None) or proj.get('data') or {}
        project_name = proj_data.get('name') or 'Projeto'

        trans = (supabase
                 .table('transcriptions')
                 .select('id,title,reuniao,created_at')
                 .eq('project_id', project_id)
                 .gte('created_at', start.isoformat())
                 .lte('created_at', end.isoformat())
                 .execute())
        trans_rows = getattr(trans, 'data', None) or trans.get('data') or []
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
        task_rows = getattr(tasks_q, 'data', None) or tasks_q.get('data') or []
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
        rows = getattr(existing, 'data', None) or existing.get('data') or []
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
                'meetings': len(meetings),
                'tasks': len(tasks),
                'next_actions': claude_result.get('next_actions', [])
            }
        }).execute())
        
        data = getattr(insert, 'data', None) or insert.get('data')
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
        supabase = get_supabase_client()
        
        query = (supabase
                .table('project_weekly_reports')
                .select('*')
                .eq('project_id', project_id)
                .order('year', desc=True)
                .order('iso_week', desc=True)
                .range(offset, offset + limit - 1))
        
        result = query.execute()
        rows = getattr(result, 'data', None) or result.get('data') or []
        
        summaries = []
        for row in rows:
            meta = row.get('meta', {})
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
        total = getattr(count_result, 'count', None) or len(summaries)
        
        return WeeklySummaryListResponse(summaries=summaries, total=total)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
