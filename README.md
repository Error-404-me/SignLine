# Speak & Sign to Text

Production-style offline Flet app for Android and desktop development. The app lives in `speak_sign_app/`; the root `pyproject.toml` is configured so `flet run` and `flet build apk` use that directory.

Read the full setup, speech model, APK build, and Bluetooth testing guide at [speak_sign_app/README.md](speak_sign_app/README.md).

Quick run:

```powershell
pip install -r .\speak_sign_app\requirements.txt
flet run .\speak_sign_app
```

Quick APK build:

```powershell
flet build apk -v
```
