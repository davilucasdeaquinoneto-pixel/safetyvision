# SafetyVision API

API FastAPI para triagem visual conservadora de riscos ocupacionais. A imagem é enviada a um modelo visual pela Hugging Face Inference Providers. O sistema aceita análises sem riscos e descarta itens sem evidência visual suficiente.

> Esta ferramenta auxilia uma triagem visual. Ela não substitui inspeção presencial nem avaliação de um profissional de Segurança do Trabalho.

## Requisitos

- Python 3.11 ou superior
- Token da Hugging Face com permissão `Make calls to Inference Providers`

## Configuração local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

No Windows PowerShell, ative o ambiente com `venv\Scripts\Activate.ps1`. As variáveis do arquivo `.env` precisam ser carregadas no terminal ou configuradas na plataforma de hospedagem.

Variáveis principais:

- `HF_TOKEN`: token secreto da Hugging Face.
- `VISION_MODEL`: modelo visual. Padrão: `Qwen/Qwen2.5-VL-3B-Instruct`.
- `FRONTEND_ORIGINS`: origens permitidas no CORS, separadas por vírgula.
- `DATABASE_PATH`: caminho do SQLite.

## Executar

```bash
uvicorn main:app --reload
```

Documentação local: `http://127.0.0.1:8000/docs`.

## Testes

```bash
python -m pytest -q
python -m compileall .
```

Os testes usam respostas simuladas e não consomem créditos.

## Render

Comando de inicialização:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Configure no serviço da API:

- `HF_TOKEN` como segredo;
- `VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct`;
- `FRONTEND_ORIGINS=https://site-projeto-integrador.onrender.com`;
- `DATABASE_PATH=data/safetyvision.db`.

O disco padrão do Render pode ser temporário. Para histórico permanente, configure um disco persistente ou outro banco.

## Privacidade e limites

- Não envie fotografias confidenciais ou com dados pessoais desnecessários.
- O plano gratuito da Hugging Face possui créditos mensais pequenos.
- A fotografia mostra apenas parte do ambiente e o modelo pode errar.
- A API nunca cria riscos simulados quando o provedor falha.
- O histórico não possui rota pública, pois o projeto ainda não tem autenticação.
