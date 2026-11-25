"use strict";
// Electron main process. Owns the window and the only thing the renderer cannot do by itself:
// talking to a real serial port. serialport is an optional dependency -- if it is missing the app
// still runs, the device list is just empty and only the simulation is available.
const { app, BrowserWindow, ipcMain, shell, Menu } = require("electron");
const path = require("path");
const net = require("net");

let SerialPort = null;
let ReadlineParser = null;
try {
  SerialPort = require("serialport").SerialPort;
  ReadlineParser = require("@serialport/parser-readline").ReadlineParser;
} catch (e) {
  SerialPort = null;
}

let win = null;
let port = null;

/* 개발·시험용 TCP 연결. 키트 프로그램을 --tcp 로 띄우고(가짜 장치 또는 ssh -L 로 끌어온 Pi)
 * PS_KIT_TCP=127.0.0.1:47800 으로 앱을 열면 장치 목록에 나옵니다. 이 PC 안의 주소만 받습니다. */
function devTcp() {
  const m = /^(127\.0\.0\.1|localhost):(\d{2,5})$/.exec(process.env.PS_KIT_TCP || "");
  return m ? { host: "127.0.0.1", port: Number(m[2]), path: "tcp://127.0.0.1:" + m[2] } : null;
}

function openTcp(t) {
  return new Promise((resolve, reject) => {
    const sock = net.createConnection({ host: t.host, port: t.port });
    let buf = "";
    let opened = false;
    sock.setEncoding("utf8");
    sock.setNoDelay(true);
    sock.once("connect", () => {
      opened = true;
      port = { write: (d) => sock.write(d), close: (cb) => { sock.destroy(); cb(); } };
      resolve(true);
    });
    sock.on("data", (d) => {
      buf += d;
      let i = buf.indexOf("\n");
      while (i >= 0) {
        send("serial:line", buf.slice(0, i).replace(/\r$/, ""));
        buf = buf.slice(i + 1);
        i = buf.indexOf("\n");
      }
    });
    sock.on("close", () => { if (opened) send("serial:closed", null); });
    sock.on("error", (err) => {
      if (opened) send("serial:error", String(err && err.message));
      else reject(err);
    });
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: "#E6EDF5",
    title: "Pendulum Studio",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
      sandbox: false
    }
  });
  Menu.setApplicationMenu(null);
  win.loadFile(path.join(__dirname, "src", "index.html"));
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
}

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

ipcMain.handle("serial:available", () => Boolean(SerialPort) || Boolean(devTcp()));

ipcMain.handle("serial:list", async () => {
  const t = devTcp();
  const extra = t ? [{ path: t.path, manufacturer: "개발용 TCP 연결", serialNumber: "", vendorId: "", productId: "" }] : [];
  if (!SerialPort) return extra;
  const ports = await SerialPort.list();
  return extra.concat(ports.map((p) => ({
    path: p.path,
    manufacturer: p.manufacturer || "",
    serialNumber: p.serialNumber || "",
    vendorId: p.vendorId || "",
    productId: p.productId || ""
  })));
});

ipcMain.handle("serial:open", async (_e, { path: devPath, baudRate }) => {
  await closePort();
  const t = devTcp();
  if (t && devPath === t.path) return openTcp(t);
  if (!SerialPort) throw new Error("serialport 모듈이 설치되지 않았습니다");
  await new Promise((resolve, reject) => {
    port = new SerialPort({ path: devPath, baudRate: baudRate || 115200 }, (err) =>
      err ? reject(err) : resolve()
    );
  });
  const parser = port.pipe(new ReadlineParser({ delimiter: "\n" }));
  parser.on("data", (line) => send("serial:line", String(line).replace(/\r$/, "")));
  port.on("close", () => send("serial:closed", null));
  port.on("error", (err) => send("serial:error", String(err && err.message)));
  return true;
});

async function closePort() {
  if (!port) return;
  const p = port;
  port = null;
  await new Promise((resolve) => p.close(() => resolve()));
}

ipcMain.handle("serial:close", async () => {
  await closePort();
  return true;
});

ipcMain.handle("serial:write", async (_e, line) => {
  if (!port) return false;
  port.write(Array.isArray(line) ? line.join("\n") + "\n" : line + "\n");
  return true;
});

app.whenReady().then(createWindow);
app.on("window-all-closed", async () => {
  await closePort();
  if (process.platform !== "darwin") app.quit();
});
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
