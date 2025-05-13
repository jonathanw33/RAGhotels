@echo off
echo ===================================
echo Removing Large File from Git Tracking
echo ===================================

echo Removing data/raw/review.txt from Git tracking...
git rm --cached data/raw/review.txt
git rm --cached data/raw/offering.txt

echo Creating a new commit with the updated .gitignore...
git add .gitignore
git commit -m "Remove large data files from Git tracking"

echo ===================================
echo Now try pushing again to GitHub:
echo git push -u origin main
echo ===================================
pause
