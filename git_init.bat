@echo off
echo ===================================
echo Initializing Git Repository
echo ===================================

echo Initializing Git...
git init

echo Adding files to staging...
git add .

echo Making initial commit...
git commit -m "Initial commit of RAG hotels project"

echo ===================================
echo Git repository initialized and initial commit created!
echo ===================================
echo.
echo If you want to push to a remote repository, use these commands:
echo git remote add origin your-repository-url
echo git branch -M main
echo git push -u origin main
echo.
pause
