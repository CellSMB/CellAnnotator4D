# Install CellAnnotator4D

Download the installer for your computer from
[GitHub Releases](https://github.com/CellSMB/CellAnnotator4D/releases).
Python is not required. These releases are unsigned and not Apple-notarized.

## Windows

1. Download the file ending in `windows-x64-setup.exe`.
2. Run it and follow the installer. It installs for your account without
   administrator privileges.
3. Launch **CellAnnotator4D** from the Start menu. If your graphics hardware
   cannot run the 3D pane, try **CellAnnotator4D (2D only)**.

Windows may show an unknown-publisher or SmartScreen warning. If you downloaded
the file from this repository and trust it, use **More info → Run anyway** when
available. An institution-managed computer may require IT approval.

To upgrade, close the app and run the newer installer. To uninstall, use
Windows Settings → Apps. Your TIFFs and annotation projects stay where you
saved them; uninstalling the application does not remove them.

## macOS

1. In **Apple menu → About This Mac**, check your chip/processor.
   Use `macos-arm64.dmg` for Apple Silicon (M-series), or `macos-x64.dmg` for Intel.
2. Open the DMG and drag **CellAnnotator4D** onto **Applications**.
3. Eject the disk image and open CellAnnotator4D from Applications.
4. If macOS blocks the app because the developer cannot be verified, open
   **System Settings → Privacy & Security** after the launch attempt and choose
   **Open Anyway**, then confirm. Only do this for a download you trust.

See [Apple's instructions](https://support.apple.com/en-au/guide/mac-help/mh40616/mac)
if your macOS version shows different wording. Managed computers may require IT
approval. Do not disable Gatekeeper globally.

To upgrade, quit the app and replace it in Applications with the newer version.
To uninstall, remove the app from Applications. Keep your TIFFs and `.cpv.json`
annotation files together when moving projects to another computer.

## First use and support

Use **File → Open TIFF** or **File → Open project…** to start. Save annotations
with **File → Save**. Projects reference the source TIFF; they do not embed it.

A compatible graphics driver/OpenGL implementation is needed for 3D. The CI
builds check 2D startup and data operations; 3D support must be checked on your
actual machine. Older operating systems may not support the bundled Qt/Python
libraries. Include your OS version, app version, and any error text when
[reporting a problem](https://github.com/CellSMB/CellAnnotator4D/issues).
