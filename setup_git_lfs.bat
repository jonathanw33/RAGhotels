@echo off
echo ===================================
echo Setting up Git LFS for Large Files
echo ===================================

REM Check if Git LFS is installed
git lfs version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Git LFS not found. Please install it first:
    echo Visit https://git-lfs.github.com/ for installation instructions
    pause
    exit /b
)

echo Initializing Git LFS...
git lfs install

echo Setting up tracking for large text files...
git lfs track "data/raw/*.txt"

echo Creating a .gitattributes file...
git add .gitattributes

echo Making a commit with LFS configuration...
git commit -m "Configure Git LFS for large text files"

echo ===================================
echo Now try pushing again to GitHub:
echo git push -u origin main
echo ===================================
echo.
echo Note: Make sure your GitHub account has enough LFS storage quota
echo.
pause
