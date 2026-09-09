#ifndef AppVersion
  #error AppVersion must be supplied by packaging/build.py
#endif
#ifndef RepoRoot
  #error RepoRoot must be supplied by packaging/build.py
#endif

[Setup]
AppId={{81DC43F3-7C66-440A-9F2E-D8A23647BF1F}
AppName=CellAnnotator4D
AppVersion={#AppVersion}
AppPublisher=CellSMB
AppPublisherURL=https://github.com/CellSMB/CellAnnotator4D
AppSupportURL=https://github.com/CellSMB/CellAnnotator4D/issues
DefaultDirName={localappdata}\Programs\CellAnnotator4D
DefaultGroupName=CellAnnotator4D
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\CellAnnotator4D.exe
OutputDir={#RepoRoot}\dist\installers
OutputBaseFilename=CellAnnotator4D-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
LicenseFile={#RepoRoot}\LICENSE
CloseApplications=yes
SetupLogging=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#RepoRoot}\dist\CellAnnotator4D\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\CellAnnotator4D"; Filename: "{app}\CellAnnotator4D.exe"
Name: "{group}\CellAnnotator4D (2D only)"; Filename: "{app}\CellAnnotator4D.exe"; Parameters: "--disable-3d"
Name: "{autodesktop}\CellAnnotator4D"; Filename: "{app}\CellAnnotator4D.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\CellAnnotator4D.exe"; Description: "Launch CellAnnotator4D"; Flags: nowait postinstall skipifsilent
