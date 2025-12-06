async function fetchSystemStats() {
    try {
        const response = await fetch("/api/system_stats");
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        const data = await response.json();
        renderSystemStats(data);
    } catch (e) {
        const container = document.getElementById("system-stats");
        if (container) {
            container.innerText = "Failed to load system stats.";
        }
    }
}

function renderSystemStats(data) {
    const container = document.getElementById("system-stats");
    if (!container) return;

    if (!data) {
        container.innerText = "No data.";
        return;
    }

    const cpuRam = data.cpu_ram || {};
    const gpus = data.gpus || [];

    let html = "";
    html += "CPU / RAM\n";
    html += "CPU: " + (cpuRam.cpu_percent ?? "?") + " %\n";
    html += "RAM: " + (cpuRam.ram_percent ?? "?") + " % (" +
        (cpuRam.ram_used_GB ?? "?") + " / " +
        (cpuRam.ram_total_GB ?? "?") + " GB)\n";
    if (cpuRam.cpu_temperature_C !== undefined) {
        html += "CPU temp: " + cpuRam.cpu_temperature_C + " °C\n";
    }

    if (gpus.length > 0) {
        html += "\nGPU\n";
        gpus.forEach(function (gpu) {
            html += "GPU " + gpu.index + " (" + gpu.name + "):\n";
            html += "  Util: " + gpu.gpu_util_percent + " %\n";
            html += "  VRAM: " + gpu.mem_used_MB + " / " +
                gpu.mem_total_MB + " MB (" + gpu.mem_util_percent + " %)\n";
            if (gpu.temperature_C !== null && gpu.temperature_C !== undefined) {
                html += "  Temp: " + gpu.temperature_C + " °C\n";
            }
        });
    } else {
        html += "\nGPU\nNo GPU stats available.";
    }

    container.textContent = html;
}

// Initial fetch + periodic refresh
fetchSystemStats();
setInterval(fetchSystemStats, 3000);

// ----- Loading overlay on form submit -----
document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const overlay = document.getElementById("loading-overlay");

    if (!form || !overlay) {
        return;
    }

    form.addEventListener("submit", function () {
        // show overlay and let the normal form submit proceed
        overlay.classList.add("visible");

        // optional: disable submit button to avoid double-click
        const submitButton = form.querySelector("button[type='submit']");
        if (submitButton) {
            submitButton.disabled = true;
        }
    });
});