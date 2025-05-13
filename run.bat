@echo off
echo ===================================
echo Hotel Recommendation RAG System
echo ===================================

if not exist hotel_rag_env\ (
    echo Creating virtual environment...
    python -m venv hotel_rag_env
    echo Virtual environment created.
)

echo Activating virtual environment...
call hotel_rag_env\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

rem Set PYTHONPATH to include the project root
set PYTHONPATH=%CD%

if "%1"=="setup" (
    echo Running setup...
    python setup.py
) else if "%1"=="process-csv" (
    echo Processing CSV data...
    python src\data_processing\process_data.py
) else if "%1"=="process-json" (
    echo Processing JSON data...
    python src\data_processing\process_json_data.py
) else if "%1"=="embeddings" (
    echo Generating embeddings...
    python src\data_processing\generate_embeddings.py
) else if "%1"=="test" (
    echo Running tests...
    python tests\test_rag.py
) else if "%1"=="test-advanced" (
    echo Running advanced search tests...
    python tests\test_advanced_search.py
) else if "%1"=="advanced" (
    echo Starting application with advanced search features...
    python src\main.py
) else if "%1"=="json" (
    echo Starting application with JSON data format...
    python src\main.py --data-format json
) else if "%1"=="csv" (
    echo Starting application with CSV data format...
    python src\main.py --data-format csv
) else (
    echo Starting the application with advanced search features...
    echo If you encounter issues, try 'run.bat setup' first.
    python src\main.py
)

pause
