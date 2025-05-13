@echo off
echo ===================================
echo Updating Libraries
echo ===================================

call hotel_rag_env\Scripts\activate
pip install --upgrade pip
pip install --upgrade llama-index==0.12.35

echo Libraries updated successfully!
pause
