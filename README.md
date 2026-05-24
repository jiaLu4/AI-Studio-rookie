# Product Image Quality Check

Static frontend and FastAPI backend for a seller-side product image quality demo.

## Frontend Local Preview

```powershell
python -m http.server 5173
```

Open `http://localhost:5173`.

## Backend Local API

Install backend dependencies:

```powershell
D:\IDES\miniconda3\python.exe -m pip install -r backend\requirements.txt
```

Run the API:

```powershell
D:\IDES\miniconda3\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then put `http://localhost:8000` into the frontend API URL field.

The default CLIP backend uses `transformers` with `openai/clip-vit-base-patch32`.
The first image-text request may download model weights. If you want to use the
original OpenCLIP route from `clip_consistency_checker.py`, install:

```powershell
D:\IDES\miniconda3\python.exe -m pip install -r backend\requirements-openclip.txt
```

Then run the API with:

```powershell
$env:CLIP_BACKEND="open_clip"
D:\IDES\miniconda3\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Vercel

Deploy only the static frontend to Vercel. The PyTorch backend should be deployed separately on a Python-capable service.
