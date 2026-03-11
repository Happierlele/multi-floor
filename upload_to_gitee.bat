@echo off
cd /d "%~dp0"
color 0A
echo ==========================================
echo    Gitee Code Upload (FORCE MODE)
echo ==========================================
echo.

echo [Step 1] Configuring...
git remote remove origin 2>nul
git remote add origin https://gitee.com/wuqile11/VLM.git
:: Force unset credential helper to ensure it asks for password in the window
git config --local credential.helper manager

echo.
echo [Step 2] Pushing code...
echo.
echo -----------------------------------------------------------
echo [INSTRUCTION]
echo 1. A window will pop up asking for username/password.
echo 2. Enter Username: wuqile11
echo 3. Enter Password.
echo.
echo NOTE: We are using FORCE PUSH to overwrite the remote README.
echo -----------------------------------------------------------
echo.

:: Pause here so user can read before it executes
pause

git push -u origin master --force

echo.
echo ==========================================
if %errorlevel% equ 0 (
    echo    SUCCESS! Upload Complete.
) else (
    echo    FAILED!
    echo    - Check your password.
    echo    - Check internet connection.
)
echo ==========================================
echo.
echo Press any key to exit...
pause >nul
