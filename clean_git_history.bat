@echo off
echo ===================================
echo Deep Cleaning Git History
echo ===================================

echo This script will completely remove the large files from your Git history.
echo WARNING: This rewrites Git history. If you've shared this repository with others,
echo they will need to re-clone it after you push the fixed version.
echo.
pause

echo Removing large files from Git history...
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch data/raw/review.txt data/raw/offering.txt" --prune-empty --tag-name-filter cat -- --all

echo Cleaning up Git repository...
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now

echo ===================================
echo Now try force-pushing to GitHub:
echo git push -f origin main
echo ===================================
echo.
echo NOTE: This is a force push and will overwrite the remote repository.
echo.
pause
