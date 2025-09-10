# LIA Resumo Semanal

Serviço de geração de resumos semanais executivos usando Anthropic Claude.

## Funcionalidades

- Geração de resumos semanais executivos
- Análise inteligente de reuniões e tarefas
- Identificação de riscos e sugestões de próximas ações
- Integração com Supabase para dados de projetos
- API REST para consumo pela plataforma Tandera

## Endpoints

- `POST /api/projects/{project_id}/weekly-summary` - Gera resumo semanal
- `GET /api/projects/{project_id}/weekly-summaries` - Lista resumos históricos
- `GET /health` - Health check

## Configuração

1. Copie `.env.example` para `.env`
2. Configure as variáveis de ambiente:
   - `SUPABASE_URL`: URL do projeto Supabase
   - `SUPABASE_KEY`: Chave anônima do Supabase
   - `SUPABASE_SERVICE_KEY`: Chave de serviço do Supabase
   - `ANTHROPIC_API_KEY`: Chave da API Anthropic Claude

## Execução

```bash
pip install -r requirements.txt
python main.py
```

O serviço estará disponível em `http://localhost:8001`

## Integração

Este serviço é consumido pela plataforma Tandera (transcription-app) para exibir resumos semanais na interface de projetos.
