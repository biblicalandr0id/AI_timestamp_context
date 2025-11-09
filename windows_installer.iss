; AI Chatbot - Windows Installer Script
; Inno Setup Script for creating professional Windows installer
; Download Inno Setup from: https://jrsoftware.org/isdl.php

#define MyAppName "AI Chatbot"
#define MyAppVersion "3.0"
#define MyAppPublisher "AI Research"
#define MyAppURL "https://github.com/your-repo/AI_timestamp_context"
#define MyAppExeName "desktop_app.py"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{AI-CHATBOT-2025}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE
InfoBeforeFile=README.md
OutputDir=installer_output
OutputBaseFilename=AI_Chatbot_Setup_{#MyAppVersion}
SetupIconFile=icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Core application files
Source: "*.py"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs
Source: "*.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "config.yaml"; DestDir: "{app}"; Flags: ignoreversion onlyifdoesntexist
Source: "chatbot.spec"; DestDir: "{app}"; Flags: ignoreversion

; Installation scripts
Source: "install.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "installer_wizard.py"; DestDir: "{app}"; Flags: ignoreversion

; Create empty directories
[Dirs]
Name: "{app}\data"
Name: "{app}\models"
Name: "{app}\plugins"
Name: "{app}\logs"
Name: "{app}\venv"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\launch_desktop.bat"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"
Name: "{group}\{#MyAppName} Web Interface"; Filename: "{app}\launch_server.bat"; WorkingDir: "{app}"
Name: "{group}\{#MyAppName} CLI"; Filename: "{app}\launch_cli.bat"; WorkingDir: "{app}"
Name: "{group}\Documentation"; Filename: "{app}\README.md"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\launch_desktop.bat"; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{app}\icon.ico"
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\launch_desktop.bat"; Tasks: quicklaunchicon; WorkingDir: "{app}"

[Run]
; Install Python dependencies
Filename: "python"; Parameters: "-m pip install --upgrade pip"; WorkingDir: "{app}"; StatusMsg: "Upgrading pip..."; Flags: runhidden
Filename: "python"; Parameters: "-m venv venv"; WorkingDir: "{app}"; StatusMsg: "Creating virtual environment..."; Flags: runhidden
Filename: "{app}\venv\Scripts\python.exe"; Parameters: "-m pip install -r requirements.txt"; WorkingDir: "{app}"; StatusMsg: "Installing Python packages (this may take 10-15 minutes)..."; Flags: runhidden
Filename: "{app}\launch_desktop.bat"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\venv"
Type: filesandordirs; Name: "{app}\data"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\*.pyc"

[Code]
var
  DependencyPage: TOutputProgressWizardPage;
  PythonInstalled: Boolean;

function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;

  // Check if Python is installed
  if not Exec('python', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if MsgBox('Python is not installed. Python 3.8 or higher is required.' + #13#10 +
              'Would you like to download Python now?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      ShellExec('open', 'https://www.python.org/downloads/', '', '', SW_SHOW, ewNoWait, ResultCode);
    end;
    Result := False;
  end
  else
  begin
    PythonInstalled := True;
  end;
end;

procedure InitializeWizard;
begin
  // Create custom page for dependency installation
  DependencyPage := CreateOutputProgressPage('Installing Dependencies',
    'Please wait while Setup installs Python dependencies...');
end;

procedure CreateLauncherScripts();
var
  DesktopScript: String;
  ServerScript: String;
  CLIScript: String;
begin
  // Desktop launcher
  DesktopScript := '@echo off' + #13#10 +
                  'cd /d "%~dp0"' + #13#10 +
                  'call venv\Scripts\activate' + #13#10 +
                  'start pythonw.exe desktop_app.py' + #13#10 +
                  'exit';
  SaveStringToFile(ExpandConstant('{app}\launch_desktop.bat'), DesktopScript, False);

  // Server launcher
  ServerScript := '@echo off' + #13#10 +
                 'cd /d "%~dp0"' + #13#10 +
                 'call venv\Scripts\activate' + #13#10 +
                 'python launch_chatbot.py server' + #13#10 +
                 'pause';
  SaveStringToFile(ExpandConstant('{app}\launch_server.bat'), ServerScript, False);

  // CLI launcher
  CLIScript := '@echo off' + #13#10 +
              'cd /d "%~dp0"' + #13#10 +
              'call venv\Scripts\activate' + #13#10 +
              'python launch_chatbot.py cli' + #13#10 +
              'pause';
  SaveStringToFile(ExpandConstant('{app}\launch_cli.bat'), CLIScript, False);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    CreateLauncherScripts();
  end;
end;

[Messages]
WelcomeLabel1=Welcome to [name] Setup
WelcomeLabel2=This will install [name/ver] on your computer.%n%n[name] is a state-of-the-art neural network chatbot with:%n%n• Real-time continual learning%n• Voice and vision capabilities%n• Interactive knowledge graphs%n• Complete privacy (100%% local)%n%nIt is recommended that you close all other applications before continuing.
FinishedHeadingLabel=Completing [name] Setup
FinishedLabel=Installation complete!%n%nTo launch AI Chatbot:%n• Click the desktop shortcut%n• Find it in your Start Menu%n• Or run launch_desktop.bat%n%nFor help, see README.md and QUICKSTART.md

[CustomMessages]
english.DependenciesInstalling=Installing Python dependencies...
english.ThisMayTakeTime=This may take 10-15 minutes depending on your internet speed.
