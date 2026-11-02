"""Offline ibm_phoenix: FakeNighthawk configuration + a Declan calibration snapshot (rzz enabled)."""
import json, os, glob, shutil, sys, warnings
from qiskit_ibm_runtime.fake_provider import FakeNighthawk
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
import qiskit_ibm_runtime.fake_provider.backends.nighthawk as nh

def make(props_path, workdir):
    os.makedirs(workdir, exist_ok=True)
    conf = json.load(open(os.path.join(os.path.dirname(nh.__file__), "conf_nighthawk.json")))
    props = json.load(open(props_path))
    names = {g["gate"] for g in props["gates"]}
    conf["basis_gates"] = sorted(set(conf["basis_gates"]) | ({"rzz", "rx"} & names))
    have = {g["name"] for g in conf.get("gates", [])}
    for gname in ("rzz", "rx"):
        if gname in names and gname not in have:
            cm = sorted({tuple(g["qubits"]) for g in props["gates"] if g["gate"] == gname})
            conf.setdefault("gates", []).append({"name": gname, "parameters": ["theta"], "qasm_def": "",
                                                 "coupling_map": [list(c) for c in cm]})
    conf["backend_name"] = "phoenix_snapshot"
    json.dump(conf, open(os.path.join(workdir, "conf_snap.json"), "w"))
    props = dict(props); props["backend_name"] = "phoenix_snapshot"
    json.dump(props, open(os.path.join(workdir, "props_snap.json"), "w"))
    class Snap(FakeBackendV2):
        dirname = workdir
        conf_filename = "conf_snap.json"
        props_filename = "props_snap.json"
        backend_name = "phoenix_snapshot"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Snap()
