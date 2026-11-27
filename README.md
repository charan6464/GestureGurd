# GestureGuard — Complete Integrated Project

This bundle includes the Flask backend, original user sign-in and random hand challenge, dedicated galaxy-themed hand-mouse game with flying clickable targets and a scrollable practice area, separate administrator portal, and administrator-only aggregate activity CSV report.

## Important compatibility note

Use **Python 3.11 (64-bit)** for this version. MediaPipe may not install correctly with Python 3.13 on Windows. If your existing `.venv` was created with Python 3.13, create a new environment using Python 3.11 rather than trying to repair that old environment.

## Windows / VS Code setup

1. Install Python 3.11 (64-bit) and ensure the Python launcher is available.
2. Extract this ZIP file.
3. Open the extracted `GestureGuard_All_In_One` folder in VS Code.
4. Open Terminal → New Terminal and run these commands from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\backend\requirements.txt
python .\backend\app.py
```

5. Open the app in your browser: <http://127.0.0.1:5000>
6. User demo account: `admin` / `ChangeMe123!`. Change the user password after logging in.
7. Administrator portal: <http://127.0.0.1:5000/admin>

## Admin credentials

The separate admin portal defaults to:

- Username: `siteadmin`
- Password: `Set-A-Strong-Admin-Password-2026!`

**Change the default admin password before using the app beyond a local demo.** In PowerShell, set your own values in the terminal before launching Flask:

```powershell
$env:GESTUREGUARD_ADMIN_USER = "your-admin-name"
$env:GESTUREGUARD_ADMIN_PASSWORD = "your-long-unique-password"
$env:GESTUREGUARD_SECRET = "replace-with-a-long-random-secret"
python .\backend\app.py
```

These environment variables apply to that terminal session. Do not share your passwords or commit them to source control.

## What the pages do

- `/` — user login, password change, and random hand challenge.
- `/mouse` — opens only after successful hand verification; shows live hand tracking, gesture instructions, clickable buttons, and a scrollable practice area.
- `/admin` — independent administrator login and activity dashboard.
- Admin dashboard — lists registered users and recent activity including successful/failed login events, hand-verification outcomes, camera starts/stops, errors, logout events, and password changes. The administrator can download a combined CSV containing user accounts and activity events.

There is no per-user activity-report download link in the user interface. The aggregate report is restricted to an authenticated admin session.

## Gestures

- Index finger: move the cursor.
- Pinch thumb and index fingertip: click.
- Closed fist: pause cursor movement.
- Open palm: resume cursor movement.

The practice area is a normal webpage area you can click and scroll. **Hand-based scrolling is not implemented**; scroll the practice area with your mouse wheel or trackpad. Cursor accuracy depends on camera placement, lighting, hand visibility, and hardware. Start with slow movements. PyAutoGUI's fail-safe is enabled; moving the cursor to the top-left corner can stop mouse movement for safety.

## Camera troubleshooting

- Close Camera, Teams, Zoom, or any other app currently using the webcam.
- Allow camera access in Windows Settings → Privacy & security → Camera.
- Run the app from the activated `.venv` terminal.
- If dependencies fail to install, confirm `python --version` shows Python 3.11.x in the activated environment.
- Do not launch multiple copies of the Flask app at the same time; they may compete for the webcam.

This is a local educational demo, not a production identity-verification or security system. Activity logs are stored locally in `gestureguard.db`, created on first run.


## Webcam preview troubleshooting
The embedded preview uses the Flask `/video_feed` MJPEG endpoint. If the camera works but the webpage stays blank, make sure the backend terminal remains open, log in, click **Start webcam**, and check the terminal for camera or MediaPipe errors. The video stream framing has been corrected in this package. Only one application should use the webcam at a time.


## Hand-based scrolling
After hand verification, point with your index finger to move the cursor. To scroll, raise your index and middle fingers in a peace sign and move your hand vertically while the cursor is over the page or practice area. Moving the hand up scrolls up; moving down scrolls down. Make a fist to pause and show an open palm to resume. Scroll speed depends on camera tracking and movement.


## Galaxy gesture game
The hand mouse page includes a galaxy-themed practice mini-game. Flying orbs move inside the arena and can be collected by hovering the hand-controlled cursor over an orb and pinching to click. Use the peace-sign scroll gesture inside the practice panel to move through the lower galaxy sectors.
