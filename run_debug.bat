@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Define variables
SET VENV_NAME=hotel_rag_env
SET DATA_FORMAT=json
SET ORG_ENGINE=src\rag\advanced_engine.py
SET DEBUG_ENGINE=src\rag\advanced_engine_debug.py
SET USE_DEBUG=false

:: Process command line arguments
IF "%1"=="debug" (
    SET USE_DEBUG=true
    SHIFT
)

IF "%1"=="json" (
    SET DATA_FORMAT=json
) ELSE IF "%1"=="csv" (
    SET DATA_FORMAT=csv
)

:: Setup environment
ECHO Starting application with %DATA_FORMAT% data format...

:: Activate virtual environment
CALL %VENV_NAME%\Scripts\activate.bat

:: Use debug version if specified
IF "%USE_DEBUG%"=="true" (
    ECHO Using enhanced debugging engine...
    
    :: Check if logs directory exists, create if not
    IF NOT EXIST "data\logs" (
        MKDIR "data\logs"
    )
    
    :: Run the application with the debug engine
    python run_app.py --data-format %DATA_FORMAT% --use-debug
) ELSE (
    :: Run the standard application
    python run_app.py --data-format %DATA_FORMAT%
)

:: Deactivate virtual environment
CALL %VENV_NAME%\Scripts\deactivate.bat

ENDLOCAL
