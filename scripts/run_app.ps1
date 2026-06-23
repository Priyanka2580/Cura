# Launches the Cura Streamlit app from the project's own .venv, first killing
# any previously running "streamlit run app.py" instance for this project.
#
# Why .venv specifically: this project's .venv has CPU-only torch +
# paddlepaddle-gpu installed (LayoutLMv3/BioBERT on CPU, PaddleOCR on GPU --
# OCR was the ~90s/image bottleneck). The global Python install still has
# GPU torch + CPU paddle, which is a different, also-valid combo but much
# slower on OCR -- don't launch via global `streamlit run app.py`, it won't
# use this venv's packages.
#
# Why kill old instances first: phase3_pipeline.py loads all models at import
# time, once per process. Closing a terminal/VS Code panel doesn't always
# kill the underlying python.exe child on Windows, so a leftover instance
# keeps holding GPU VRAM -- stack a few of these and the 6GB RTX 4050 runs
# out of memory, which is why the app "breaks after the first image" until
# everything is force-restarted. Always launch through this script instead
# of running streamlit directly.

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*streamlit.exe*run*app.py*" } |
    ForEach-Object {
        Write-Host "Stopping previous instance (PID $($_.ProcessId))"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Start-Sleep -Seconds 2

Set-Location $PSScriptRoot\..
& .\.venv\Scripts\python.exe -m streamlit run app.py
