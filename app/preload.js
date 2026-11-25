"use strict";
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("studio", {
  serial: {
    available: () => ipcRenderer.invoke("serial:available"),
    list: () => ipcRenderer.invoke("serial:list"),
    open: (path, baudRate) => ipcRenderer.invoke("serial:open", { path, baudRate }),
    close: () => ipcRenderer.invoke("serial:close"),
    write: (line) => ipcRenderer.invoke("serial:write", line),
    onLine: (fn) => ipcRenderer.on("serial:line", (_e, line) => fn(line)),
    onClosed: (fn) => ipcRenderer.on("serial:closed", () => fn()),
    onError: (fn) => ipcRenderer.on("serial:error", (_e, msg) => fn(msg))
  }
});
