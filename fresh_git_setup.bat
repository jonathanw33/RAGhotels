@echo off
echo ===================================
echo Fresh Git Repository Setup
echo ===================================

echo This script will:
echo 1. Backup the old Git repository
echo 2. Create a new Git repository
echo 3. Add and commit all files (except those in .gitignore)
echo 4. Force push to GitHub
echo.
echo Press any key to continue...
pause > nul

echo.
echo Step 1: Backing up old Git repository...
if exist .git (
    ren .git .git_old
    echo Old repository backed up as .git_old
) else (
    echo No existing .git directory found
)

echo.
echo Step 2: Creating new Git repository...
git init
git remote add origin https://github.com/jonathanw33/RAGhotels.git

echo.
echo Step 3: Adding and committing files...
git add .
git commit -m "Initial commit of RAG hotels project"

echo.
echo Step 4: Setting up main branch and pushing...
git branch -M main
echo.
echo Ready to push to GitHub. This will force push and replace any content on GitHub.
echo.
echo Press any key to continue...
pause > nul

git push -f origin main

echo.
echo ===================================
echo Process completed!
echo ===================================
echo.
echo If you see any errors about large files, make sure:
echo 1. Your .gitignore file contains these lines:
echo    data/raw/review.txt
echo    data/raw/offering.txt
echo 2. These files actually exist in your directory structure
echo.
pause
