"""AMD GPU stats in nvidia-smi's CSV layout, for the UI's system monitor.

Prints one line per GPU in the column order of the UI's nvidia-smi query
(index, name, driver, temp, util, mem util, mem total/free/used MiB, power
draw/limit W, graphics/memory clock MHz, fan %), so ui/src/server/monitor.ts
parses both the same way. Linux reads the amdgpu sysfs files, Windows goes
through AMD's ADLX bindings (`amd-adlx`). Indices follow HIP's device order
so they line up with HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES.

Run directly by path (no toolkit imports): `python amd_gpu_stats.py [--loop-ms N]`.
"""

import glob
import os
import re
import subprocess
import sys
import threading
import time

AMD_VENDOR = "0x1002"


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _num(path, scale=1.0):
    value = _read(path)
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- Linux


PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids", "/usr/share/pci.ids")
DRM_CARDS = "/sys/class/drm/card[0-9]*/device"
KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _kfd_pci_order():
    """(domain, location_id) of each KFD GPU node, in node order (= HIP device order)."""
    order = []
    try:
        nodes = sorted(os.listdir(KFD_NODES), key=lambda n: int(n) if n.isdigit() else 0)
    except OSError:
        return order
    for node in nodes:
        props = {}
        for line in (_read(os.path.join(KFD_NODES, node, "properties")) or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                props[parts[0]] = int(parts[1])
        # CPU nodes have no SIMDs
        if props.get("simd_count") and "location_id" in props:
            order.append((props.get("domain", 0), props["location_id"]))
    return order


def _pci_location(card):
    """(domain, location_id) from a PCI address dir name like 0000:03:00.0."""
    m = re.match(r"([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-7])$", os.path.basename(card), re.I)
    if not m:
        return None
    domain, bus, dev, func = (int(x, 16) for x in m.groups())
    # KFD packs location_id the same way as the PCI BDF
    return domain, (bus << 8) | (dev << 3) | func


def _pci_names(device_ids):
    """{device_id: marketing name} from the system pci.ids database."""
    names = {}
    for path in PCI_IDS:
        try:
            f = open(path, encoding="utf-8", errors="replace")
        except OSError:
            continue
        with f:
            in_amd = False
            for line in f:
                if not line.startswith("\t"):
                    in_amd = line.startswith("1002 ")
                elif in_amd and not line.startswith("\t\t"):
                    dev, _, name = line.strip().partition(" ")
                    if dev in device_ids:
                        names[dev] = name.strip()
        break
    return names


class LinuxSource(object):
    def __init__(self):
        cards = []
        for dev in glob.glob(DRM_CARDS):
            if _read(os.path.join(dev, "vendor")) != AMD_VENDOR:
                continue
            if _num(os.path.join(dev, "mem_info_vram_total")) is None:
                continue
            cards.append(os.path.realpath(dev))
        # HIP numbers GPUs in KFD node order, which is usually but not always
        # PCI order; cards KFD does not list go last, in PCI order
        kfd = _kfd_pci_order()

        def hip_order(card):
            loc = _pci_location(card)
            return (kfd.index(loc) if loc in kfd else len(kfd), os.path.basename(card))

        self.cards = sorted(set(cards), key=hip_order)
        ids = {(_read(os.path.join(c, "device")) or "")[2:] for c in self.cards}
        pci = _pci_names(ids)
        self.names = []
        for c in self.cards:
            dev_id = (_read(os.path.join(c, "device")) or "")[2:]
            name = _read(os.path.join(c, "product_name")) or pci.get(dev_id)
            self.names.append(name or "AMD GPU (%s)" % dev_id)
        self.driver = _read("/sys/module/amdgpu/version") or "amdgpu"

    def sample(self):
        rows = []
        for index, (card, name) in enumerate(zip(self.cards, self.names)):
            hwmon = (glob.glob(os.path.join(card, "hwmon", "hwmon*")) or [""])[0]
            total = _num(os.path.join(card, "mem_info_vram_total"), 1.0 / 2**20) or 0
            used = _num(os.path.join(card, "mem_info_vram_used"), 1.0 / 2**20) or 0
            power = _num(os.path.join(hwmon, "power1_average"), 1e-6)
            if power is None:
                power = _num(os.path.join(hwmon, "power1_input"), 1e-6)
            pwm = _num(os.path.join(hwmon, "pwm1"))
            pwm_max = _num(os.path.join(hwmon, "pwm1_max")) or 255
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "driver": self.driver,
                    "temp": _num(os.path.join(hwmon, "temp1_input"), 1e-3),
                    "util": _num(os.path.join(card, "gpu_busy_percent")),
                    "mem_util": _num(os.path.join(card, "mem_busy_percent")),
                    "total": total,
                    "used": used,
                    "power": power,
                    "power_limit": _num(os.path.join(hwmon, "power1_cap"), 1e-6),
                    "sclk": _num(os.path.join(hwmon, "freq1_input"), 1e-6),
                    "mclk": _num(os.path.join(hwmon, "freq2_input"), 1e-6),
                    "fan": None if pwm is None else pwm * 100.0 / pwm_max,
                }
            )
        return rows


# ---------------------------------------------------------------- Windows


def _hip_bus_order():
    """PCI bus numbers in HIP device order, from the hipInfo the ROCm wheels ship."""
    exe = os.path.join(os.path.dirname(sys.executable), "hipInfo.exe")
    if not os.path.isfile(exe):
        return None
    try:
        out = subprocess.run(
            [exe],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.decode("utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [int(b) for b in re.findall(r"^pciBusID:\s*(\d+)", out, re.M)] or None


class WindowsSource(object):
    def __init__(self):
        from adlx import ADLX

        self.helper = ADLX.ADLXHelper()
        if self.helper.Initialize() != ADLX.ADLX_RESULT.ADLX_OK:
            raise RuntimeError("ADLX failed to initialize")
        system = self.helper.GetSystemServices()
        self.perf = system.GetPerformanceMonitoringServices()
        gpus = list(system.GetGPUs())
        # UniqueId packs the PCI location as (bus << 8) | (dev << 3) | func
        by_bus = {(g.UniqueId() >> 8) & 0xFF: g for g in gpus}
        order = _hip_bus_order()
        if order:
            self.gpus = [by_bus[b] for b in order if b in by_bus]
        else:
            self.gpus = [by_bus[b] for b in sorted(by_bus)]
        self.static = []
        for g in self.gpus:
            support = self.perf.GetSupportedGPUMetrics(g)
            power_limit = None
            if support.IsSupportedGPUTotalBoardPower():
                power_limit = support.GetGPUTotalBoardPowerRange()[1]
            elif support.IsSupportedGPUPower():
                power_limit = support.GetGPUPowerRange()[1]
            self.static.append(
                {
                    "name": g.Name(),
                    "driver": g.AMDSoftwareVersion(),
                    "total": float(g.TotalVRAM()),
                    "support": support,
                    "power_limit": power_limit,
                }
            )

    def _metric(self, support, metrics, name):
        try:
            if getattr(support, "IsSupported" + name)():
                return float(getattr(metrics, name)())
        except Exception:
            pass
        return None

    def sample(self):
        rows = []
        for index, (g, st) in enumerate(zip(self.gpus, self.static)):
            support = st["support"]
            metrics = self.perf.GetCurrentGPUMetrics(g)
            # board power includes VRAM and VRM losses, GPUPower is chip only
            power = self._metric(support, metrics, "GPUTotalBoardPower")
            if power is None:
                power = self._metric(support, metrics, "GPUPower")
            rows.append(
                {
                    "index": index,
                    "name": st["name"],
                    "driver": st["driver"],
                    "temp": self._metric(support, metrics, "GPUTemperature"),
                    "util": self._metric(support, metrics, "GPUUsage"),
                    "mem_util": None,
                    "total": st["total"],
                    "used": self._metric(support, metrics, "GPUVRAM") or 0,
                    "power": power,
                    "power_limit": st["power_limit"],
                    "sclk": self._metric(support, metrics, "GPUClockSpeed"),
                    "mclk": self._metric(support, metrics, "GPUVRAMClockSpeed"),
                    "fan": self._metric(support, metrics, "GPUFanDuty"),
                }
            )
            del metrics
        return rows

    def close(self):
        # ADLX crashes at interpreter exit unless every interface is released
        # before Terminate
        self.static = self.gpus = self.perf = None
        self.helper.Terminate()


# ---------------------------------------------------------------- output


def _fmt(value, digits=0):
    if value is None:
        return "[N/A]"
    return "%.*f" % (digits, value)


def format_row(row):
    total = row["total"]
    used = min(row["used"], total)
    fields = [
        str(row["index"]),
        row["name"].replace(",", " "),
        str(row["driver"]).replace(",", " "),
        _fmt(row["temp"]),
        _fmt(row["util"]),
        _fmt(row["mem_util"]),
        _fmt(total),
        _fmt(total - used),
        _fmt(used),
        _fmt(row["power"], 2),
        _fmt(row["power_limit"], 2),
        _fmt(row["sclk"]),
        _fmt(row["mclk"]),
        _fmt(row["fan"]),
    ]
    return ", ".join(fields)


def _exit_with_parent():
    # the UI holds our stdin open; EOF means it went away (even if killed hard)
    try:
        sys.stdin.read()
    except Exception:
        pass
    os._exit(0)


def main():
    loop_ms = None
    if "--loop-ms" in sys.argv:
        loop_ms = int(sys.argv[sys.argv.index("--loop-ms") + 1])
    source = WindowsSource() if sys.platform == "win32" else LinuxSource()
    if loop_ms:
        threading.Thread(target=_exit_with_parent, daemon=True).start()
    try:
        while True:
            lines = [format_row(r) for r in source.sample()]
            if not lines:
                # no AMD GPU: the UI stops polling on a silent exit
                sys.exit(2)
            sys.stdout.write("".join(line + "\n" for line in lines))
            sys.stdout.flush()
            if not loop_ms:
                break
            time.sleep(loop_ms / 1000.0)
    finally:
        if hasattr(source, "close"):
            source.close()


if __name__ == "__main__":
    main()
