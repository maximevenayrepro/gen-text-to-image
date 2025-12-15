// Data history buffers (circular, max 50 points)
const MAX_HISTORY = 50;
const history = {
    cpu: [],
    ram: [],
    gpus: {},
    gpuMem: {},
    gpuSharedMem: {}
};

// Save history to localStorage
function saveHistoryToStorage() {
    try {
        localStorage.setItem('systemStatsHistory', JSON.stringify(history));
    } catch (e) {
        console.warn("Failed to save history to localStorage:", e);
    }
}

// Load history from localStorage
function loadHistoryFromStorage() {
    try {
        const saved = localStorage.getItem('systemStatsHistory');
        if (saved) {
            const parsed = JSON.parse(saved);
            // Merge saved history with current structure
            if (parsed.cpu) history.cpu = parsed.cpu.slice(-MAX_HISTORY);
            if (parsed.ram) history.ram = parsed.ram.slice(-MAX_HISTORY);
            if (parsed.gpus) Object.assign(history.gpus, parsed.gpus);
            if (parsed.gpuMem) Object.assign(history.gpuMem, parsed.gpuMem);
            if (parsed.gpuSharedMem) Object.assign(history.gpuSharedMem, parsed.gpuSharedMem);
        }
    } catch (e) {
        console.warn("Failed to load history from localStorage:", e);
    }
}

// Chart instances
let charts = {
    cpu: null,
    ram: null,
    gpus: {},
    gpuMem: {},
    gpuSharedMem: {}
};

// Chart.js configuration
const chartConfig = {
    type: 'line',
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                titleColor: '#e5e7eb',
                bodyColor: '#9ca3af',
                borderColor: 'rgba(59, 130, 246, 0.5)',
                borderWidth: 1,
                padding: 8,
                titleFont: { size: 11 },
                bodyFont: { size: 10 }
            }
        },
        scales: {
            x: {
                display: false
            },
            y: {
                display: false,
                min: 0,
                max: 100,
                suggestedMin: 0,
                suggestedMax: 100,
                ticks: {
                    min: 0,
                    max: 100
                }
            }
        },
        elements: {
            point: {
                radius: 0,
                hoverRadius: 4
            },
            line: {
                borderWidth: 2,
                tension: 0.4
            }
        }
    }
};

function createChart(canvasId, color, label) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    return new Chart(ctx, {
        ...chartConfig,
        data: {
            labels: [],
            datasets: [{
                label: label,
                data: [],
                borderColor: color,
                backgroundColor: color + '20',
                fill: true
            }]
        }
    });
}

function addToHistory(buffer, value, maxSize) {
    buffer.push(value);
    if (buffer.length > maxSize) {
        buffer.shift();
    }
    // Save to localStorage after each update
    saveHistoryToStorage();
}

function updateChart(chart, data, labels) {
    if (!chart) return;
    chart.data.datasets[0].data = data;
    chart.data.labels = labels;
    // Force Y axis to always be 0-100
    if (chart.options.scales && chart.options.scales.y) {
        chart.options.scales.y.min = 0;
        chart.options.scales.y.max = 100;
    }
    chart.update('none');
}

function ensureGPUContainer(gpuIndex, type) {
    const gpuKey = "gpu-" + gpuIndex;
    const containerId = gpuKey + "-" + type + "-container";
    let containerEl = document.getElementById(containerId);
    
    if (!containerEl) {
        const metricsGrid = document.querySelector(".stats-metrics-grid");
        if (!metricsGrid) return null;
        
        containerEl = document.createElement("div");
        containerEl.className = "stats-metric-item";
        containerEl.id = containerId;
        containerEl.style.display = "block";
        
        const labelEl = document.createElement("div");
        labelEl.className = "stats-metric-label";
        labelEl.id = gpuKey + "-" + type + "-label";
        
        const valueEl = document.createElement("div");
        valueEl.className = "stats-metric-value";
        valueEl.id = gpuKey + "-" + type + "-value";
        valueEl.textContent = "--";
        
        const canvasEl = document.createElement("canvas");
        canvasEl.id = gpuKey + "-" + type + "-chart";
        
        containerEl.appendChild(labelEl);
        containerEl.appendChild(valueEl);
        containerEl.appendChild(canvasEl);
        
        metricsGrid.appendChild(containerEl);
    }
    
    return containerEl;
}

async function fetchSystemStats() {
    try {
        const response = await fetch("/api/system_stats");
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        const data = await response.json();
        renderSystemStats(data);
    } catch (e) {
        console.error("Failed to fetch system stats:", e);
    }
}

function renderSystemStats(data) {
    if (!data) return;

    const cpuRam = data.cpu_ram || {};
    const gpus = data.gpus || [];
    
    // Restore history from localStorage on first render
    if (history.cpu.length === 0 && history.ram.length === 0) {
        loadHistoryFromStorage();
    }

    // Update CPU chart
    const cpuPercent = cpuRam.cpu_percent ?? 0;
    addToHistory(history.cpu, cpuPercent, MAX_HISTORY);
    const cpuValueEl = document.getElementById("cpu-value");
    if (cpuValueEl) {
        cpuValueEl.textContent = cpuPercent.toFixed(0) + "%";
    }
    if (!charts.cpu) {
        charts.cpu = createChart("cpu-chart", "rgba(59, 130, 246, 0.8)", "CPU");
    }
    if (charts.cpu) {
        const labels = history.cpu.map((_, i) => "");
        updateChart(charts.cpu, history.cpu, labels);
    }

    // Update RAM chart
    const ramPercent = cpuRam.ram_percent ?? 0;
    addToHistory(history.ram, ramPercent, MAX_HISTORY);
    const ramValueEl = document.getElementById("ram-value");
    if (ramValueEl) {
        const ramUsed = cpuRam.ram_used_GB ?? 0;
        const ramTotal = cpuRam.ram_total_GB ?? 0;
        ramValueEl.textContent = ramPercent.toFixed(0) + "% (" + ramUsed.toFixed(1) + "/" + ramTotal.toFixed(1) + " GB)";
    }
    if (!charts.ram) {
        charts.ram = createChart("ram-chart", "rgba(34, 197, 94, 0.8)", "RAM");
    }
    if (charts.ram) {
        const labels = history.ram.map((_, i) => "");
        updateChart(charts.ram, history.ram, labels);
    }

    // Update GPU charts
    gpus.forEach(function (gpu) {
        const gpuIndex = gpu.index;
        const gpuKey = "gpu-" + gpuIndex;
        
        // Initialize history buffers
        if (!history.gpus[gpuKey]) {
            history.gpus[gpuKey] = [];
        }
        if (!history.gpuMem[gpuKey]) {
            history.gpuMem[gpuKey] = [];
        }
        if (!history.gpuSharedMem[gpuKey]) {
            history.gpuSharedMem[gpuKey] = [];
        }

        const shortName = gpu.name.length > 15 ? gpu.name.substring(0, 15) + "..." : gpu.name;
        
        // GPU Usage chart
        const gpuUtil = gpu.gpu_util_percent ?? 0;
        addToHistory(history.gpus[gpuKey], gpuUtil, MAX_HISTORY);
        ensureGPUContainer(gpuIndex, "usage");
        
        const usageLabelEl = document.getElementById(gpuKey + "-usage-label");
        const usageValueEl = document.getElementById(gpuKey + "-usage-value");
        
        if (usageLabelEl) {
            usageLabelEl.textContent = "GPU " + gpuIndex + " Usage";
        }
        if (usageValueEl) {
            usageValueEl.textContent = gpuUtil.toFixed(0) + "%";
        }

        if (!charts.gpus[gpuKey]) {
            charts.gpus[gpuKey] = createChart(gpuKey + "-usage-chart", "rgba(168, 85, 247, 0.8)", "GPU " + gpuIndex + " Usage");
        }
        if (charts.gpus[gpuKey]) {
            const labels = history.gpus[gpuKey].map((_, i) => "");
            updateChart(charts.gpus[gpuKey], history.gpus[gpuKey], labels);
        }

        // GPU Memory (VRAM) chart
        const gpuMemPercent = gpu.mem_util_percent ?? 0;
        addToHistory(history.gpuMem[gpuKey], gpuMemPercent, MAX_HISTORY);
        ensureGPUContainer(gpuIndex, "mem");
        
        const memLabelEl = document.getElementById(gpuKey + "-mem-label");
        const memValueEl = document.getElementById(gpuKey + "-mem-value");
        
        if (memLabelEl) {
            memLabelEl.textContent = "GPU " + gpuIndex + " Memory";
        }
        if (memValueEl) {
            const memUsed = gpu.mem_used_MB ?? 0;
            const memTotal = gpu.mem_total_MB ?? 0;
            memValueEl.textContent = gpuMemPercent.toFixed(0) + "% (" + memUsed.toFixed(0) + "/" + memTotal.toFixed(0) + " MB)";
        }

        if (!charts.gpuMem[gpuKey]) {
            charts.gpuMem[gpuKey] = createChart(gpuKey + "-mem-chart", "rgba(236, 72, 153, 0.8)", "GPU " + gpuIndex + " Memory");
        }
        if (charts.gpuMem[gpuKey]) {
            const labels = history.gpuMem[gpuKey].map((_, i) => "");
            updateChart(charts.gpuMem[gpuKey], history.gpuMem[gpuKey], labels);
        }

        // GPU Shared Memory chart (using mem_util_percent like GPU Memory)
        const gpuSharedMemPercent = gpu.mem_util_percent ?? 0;
        addToHistory(history.gpuSharedMem[gpuKey], gpuSharedMemPercent, MAX_HISTORY);
        ensureGPUContainer(gpuIndex, "shared");
        
        const sharedLabelEl = document.getElementById(gpuKey + "-shared-label");
        const sharedValueEl = document.getElementById(gpuKey + "-shared-value");
        
        if (sharedLabelEl) {
            sharedLabelEl.textContent = "GPU " + gpuIndex + " Shared Mem";
        }
        if (sharedValueEl) {
            const memUsed = gpu.mem_used_MB ?? 0;
            const memTotal = gpu.mem_total_MB ?? 0;
            sharedValueEl.textContent = gpuSharedMemPercent.toFixed(0) + "% (" + memUsed.toFixed(0) + "/" + memTotal.toFixed(0) + " MB)";
        }

        if (!charts.gpuSharedMem[gpuKey]) {
            charts.gpuSharedMem[gpuKey] = createChart(gpuKey + "-shared-chart", "rgba(251, 146, 60, 0.8)", "GPU " + gpuIndex + " Shared Mem");
        }
        if (charts.gpuSharedMem[gpuKey]) {
            const labels = history.gpuSharedMem[gpuKey].map((_, i) => "");
            updateChart(charts.gpuSharedMem[gpuKey], history.gpuSharedMem[gpuKey], labels);
        }
    });
}

// Initialize charts on DOM ready
document.addEventListener("DOMContentLoaded", function () {
    // Charts will be created on first data fetch
    
    // Add Model Modal functionality
    const addModelBtn = document.getElementById("add-model-btn");
    const modal = document.getElementById("add-model-modal");
    const modalCloseBtn = document.getElementById("modal-close-btn");
    const modalCancelBtn = document.getElementById("modal-cancel-btn");
    const modalOkBtn = document.getElementById("modal-ok-btn");
    const modalUrlInput = document.getElementById("modal-model-url");
    const customModelRepoInput = document.getElementById("custom_model_repo");
    
    function openModal() {
        if (modal) {
            modal.style.display = "flex";
            if (modalUrlInput) {
                modalUrlInput.focus();
            }
        }
    }
    
    function closeModal() {
        if (modal) {
            modal.style.display = "none";
            if (modalUrlInput) {
                modalUrlInput.value = "";
            }
        }
    }
    
    if (addModelBtn) {
        addModelBtn.addEventListener("click", openModal);
    }
    
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener("click", closeModal);
    }
    
    if (modalCancelBtn) {
        modalCancelBtn.addEventListener("click", closeModal);
    }
    
    if (modalOkBtn && modalUrlInput && customModelRepoInput) {
        modalOkBtn.addEventListener("click", function () {
            const url = modalUrlInput.value.trim();
            if (url) {
                customModelRepoInput.value = url;
                closeModal();
                // Optionally submit the form or just let user click Generate
            } else {
                alert("Please enter a model URL or repo_id");
            }
        });
    }
    
    // Close modal when clicking outside
    if (modal) {
        modal.addEventListener("click", function (e) {
            if (e.target === modal) {
                closeModal();
            }
        });
    }
    
    // Close modal with Escape key
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape" && modal && modal.style.display === "flex") {
            closeModal();
        }
    });
    
    // Prompt functionality
    let prompts = [];
    const randomPromptBtn = document.getElementById("random-prompt-btn");
    const clearPromptBtn = document.getElementById("clear-prompt-btn");
    const promptTextarea = document.getElementById("prompt");
    
    // Clear prompt button
    if (clearPromptBtn && promptTextarea) {
        clearPromptBtn.addEventListener("click", function () {
            promptTextarea.value = "";
            promptTextarea.focus();
            promptTextarea.dispatchEvent(new Event("input", { bubbles: true }));
        });
    }
    
    if (randomPromptBtn && promptTextarea) {
        // Load prompts from JSON file
        fetch("/static/prompts.json")
            .then(response => {
                if (!response.ok) {
                    throw new Error("Failed to load prompts");
                }
                return response.json();
            })
            .then(data => {
                prompts = data;
            })
            .catch(error => {
                console.error("Error loading prompts:", error);
            });
        
        // Handle dice button click
        randomPromptBtn.addEventListener("click", function () {
            if (prompts.length === 0) {
                console.warn("Prompts not loaded yet");
                return;
            }
            
            // Get random prompt
            const randomIndex = Math.floor(Math.random() * prompts.length);
            const randomPrompt = prompts[randomIndex];
            
            // Fill textarea with random prompt
            promptTextarea.value = randomPrompt;
            
            // Optional: trigger input event for any listeners
            promptTextarea.dispatchEvent(new Event("input", { bubbles: true }));
        });
    }
});

// Load history from localStorage on page load
loadHistoryFromStorage();

// Restore charts with saved history on page load
function restoreChartsFromHistory() {
    // Restore CPU chart
    if (history.cpu.length > 0) {
        if (!charts.cpu) {
            charts.cpu = createChart("cpu-chart", "rgba(59, 130, 246, 0.8)", "CPU");
        }
        if (charts.cpu) {
            const labels = history.cpu.map((_, i) => "");
            updateChart(charts.cpu, history.cpu, labels);
        }
    }
    
    // Restore RAM chart
    if (history.ram.length > 0) {
        if (!charts.ram) {
            charts.ram = createChart("ram-chart", "rgba(34, 197, 94, 0.8)", "RAM");
        }
        if (charts.ram) {
            const labels = history.ram.map((_, i) => "");
            updateChart(charts.ram, history.ram, labels);
        }
    }
    
    // Restore GPU charts
    Object.keys(history.gpus).forEach(function(gpuKey) {
        const gpuIndex = gpuKey.replace("gpu-", "");
        if (history.gpus[gpuKey].length > 0) {
            ensureGPUContainer(parseInt(gpuIndex), "usage");
            if (!charts.gpus[gpuKey]) {
                charts.gpus[gpuKey] = createChart(gpuKey + "-usage-chart", "rgba(168, 85, 247, 0.8)", "GPU " + gpuIndex + " Usage");
            }
            if (charts.gpus[gpuKey]) {
                const labels = history.gpus[gpuKey].map((_, i) => "");
                updateChart(charts.gpus[gpuKey], history.gpus[gpuKey], labels);
            }
        }
    });
    
    Object.keys(history.gpuMem).forEach(function(gpuKey) {
        const gpuIndex = gpuKey.replace("gpu-", "");
        if (history.gpuMem[gpuKey].length > 0) {
            ensureGPUContainer(parseInt(gpuIndex), "mem");
            if (!charts.gpuMem[gpuKey]) {
                charts.gpuMem[gpuKey] = createChart(gpuKey + "-mem-chart", "rgba(236, 72, 153, 0.8)", "GPU " + gpuIndex + " Memory");
            }
            if (charts.gpuMem[gpuKey]) {
                const labels = history.gpuMem[gpuKey].map((_, i) => "");
                updateChart(charts.gpuMem[gpuKey], history.gpuMem[gpuKey], labels);
            }
        }
    });
    
    Object.keys(history.gpuSharedMem).forEach(function(gpuKey) {
        const gpuIndex = gpuKey.replace("gpu-", "");
        if (history.gpuSharedMem[gpuKey].length > 0) {
            ensureGPUContainer(parseInt(gpuIndex), "shared");
            if (!charts.gpuSharedMem[gpuKey]) {
                charts.gpuSharedMem[gpuKey] = createChart(gpuKey + "-shared-chart", "rgba(251, 146, 60, 0.8)", "GPU " + gpuIndex + " Shared Mem");
            }
            if (charts.gpuSharedMem[gpuKey]) {
                const labels = history.gpuSharedMem[gpuKey].map((_, i) => "");
                updateChart(charts.gpuSharedMem[gpuKey], history.gpuSharedMem[gpuKey], labels);
            }
        }
    });
}

// Load history and restore charts on page load
document.addEventListener("DOMContentLoaded", function() {
    loadHistoryFromStorage();
    // Wait a bit for DOM to be ready, then restore charts
    setTimeout(restoreChartsFromHistory, 100);
});

// Initial fetch + periodic refresh
fetchSystemStats();
setInterval(fetchSystemStats, 3000);

// ----- Update dimensions and steps when model changes -----
document.addEventListener("DOMContentLoaded", function () {
    const modelSelect = document.getElementById("model_name");
    const heightInput = document.getElementById("height");
    const widthInput = document.getElementById("width");
    const stepsInput = document.getElementById("steps");

    if (!modelSelect || !heightInput || !widthInput || !stepsInput) {
        return;
    }

    async function updateDefaultsForModel(modelName) {
        try {
            const response = await fetch(`/api/model_info/${encodeURIComponent(modelName)}`);
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            const data = await response.json();
            
            if (data.default_dimensions) {
                heightInput.value = data.default_dimensions.height;
                widthInput.value = data.default_dimensions.width;
            }
            
            if (data.default_steps) {
                stepsInput.value = data.default_steps;
            }
        } catch (e) {
            console.error("Failed to fetch model defaults:", e);
            // Keep current values on error
        }
    }

    modelSelect.addEventListener("change", function () {
        const selectedModel = modelSelect.value;
        if (selectedModel) {
            updateDefaultsForModel(selectedModel);
        }
    });
});

// Function to display generated images
function displayGeneratedImages(images) {
    console.log("displayGeneratedImages called with", images.length, "images");
    let galleryGrid = document.querySelector(".gallery-grid");
    const emptyState = document.querySelector(".empty-state");
    const galleryPanel = document.querySelector(".gallery-panel");
    
    // Create gallery-grid if it doesn't exist
    if (!galleryGrid && galleryPanel) {
        galleryGrid = document.createElement("div");
        galleryGrid.className = "gallery-grid";
        galleryPanel.appendChild(galleryGrid);
    }
    
    if (!galleryGrid) {
        console.error("Could not find or create gallery-grid");
        return;
    }
    
    // Hide empty state
    if (emptyState) {
        emptyState.style.display = "none";
    }
    
    // Clear existing images (optional - or append)
    galleryGrid.innerHTML = "";
    galleryGrid.style.display = "grid";
    
    // Create image cards for each generated image
    images.forEach(function(img) {
        if (!img.data_url) {
            console.error("Image missing data_url:", img);
            return;
        }
        
        const imageCard = document.createElement("div");
        imageCard.className = "image-card";
        
        imageCard.innerHTML = `
            <div class="image-wrapper">
                <img src="${img.data_url}" alt="Generated image">
                <div class="image-overlay"></div>
                <div class="image-actions">
                    <form method="post" action="/generate" style="margin:0;">
                        <input type="hidden" name="model_name" value="${escapeHtml(img.model_name)}">
                        <input type="hidden" name="custom_model_repo" value="">
                        <input type="hidden" name="prompt" value="${escapeHtml(img.prompt)}">
                        <input type="hidden" name="steps" value="${img.steps}">
                        <input type="hidden" name="height" value="${img.height}">
                        <input type="hidden" name="width" value="${img.width}">
                        <input type="hidden" name="num_images" value="1">
                        <button type="submit" class="icon-button" title="Regenerate">
                            <i class="fa-solid fa-rotate-right"></i>
                        </button>
                    </form>
                    <a href="${img.view_url}" target="_blank" class="icon-link" title="Open in new tab">
                        <i class="fa-solid fa-up-right-from-square"></i>
                    </a>
                    <a href="${img.download_url}" class="icon-link" title="Download">
                        <i class="fa-solid fa-download"></i>
                    </a>
                </div>
            </div>
        `;
        
        galleryGrid.appendChild(imageCard);
    });
    
    console.log("Images displayed, gallery-grid now has", galleryGrid.children.length, "children");
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ----- Form submit handling with AJAX and cancellation -----
document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const generateBtn = document.getElementById("generate-btn");
    const cancelBtn = document.getElementById("cancel-btn");
    const generatingPlaceholder = document.getElementById("generating-placeholder");
    const galleryGrid = document.querySelector(".gallery-grid");
    const emptyState = document.querySelector(".empty-state");
    let abortController = null;

    if (!form || !generateBtn || !cancelBtn || !generatingPlaceholder) {
        return;
    }

    form.addEventListener("submit", async function (e) {
        e.preventDefault(); // Prevent default form submission
        
        // Disable generate button and show cancel button
        generateBtn.disabled = true;
        cancelBtn.style.display = "inline-flex";
        
        // Hide empty state and show generating placeholder
        if (emptyState) {
            emptyState.style.display = "none";
        }
        if (galleryGrid) {
            galleryGrid.style.display = "none";
        }
        generatingPlaceholder.style.display = "block";
        
        // Create AbortController for cancellation
        abortController = new AbortController();
        
        // Get form data
        const formData = new FormData(form);
        const data = {
            model_name: formData.get("model_name"),
            custom_model_repo: formData.get("custom_model_repo") || "",
            prompt: formData.get("prompt"),
            steps: parseInt(formData.get("steps")),
            height: parseInt(formData.get("height")),
            width: parseInt(formData.get("width")),
            num_images: parseInt(formData.get("num_images"))
        };
        
        try {
            const response = await fetch("/api/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(data),
                signal: abortController.signal
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Generation failed");
            }
            
            const result = await response.json();
            
            if (result.success && result.images && result.images.length > 0) {
                // Display images directly without reloading
                console.log("Displaying images:", result.images.length);
                displayGeneratedImages(result.images);
                
                // Reset UI state
                generateBtn.disabled = false;
                cancelBtn.style.display = "none";
                generatingPlaceholder.style.display = "none";
            } else {
                throw new Error("Invalid response from server or no images generated");
            }
        } catch (error) {
            if (error.name === "AbortError") {
                console.log("Generation cancelled");
                // Reset UI state
                generateBtn.disabled = false;
                cancelBtn.style.display = "none";
                generatingPlaceholder.style.display = "none";
                if (emptyState) {
                    emptyState.style.display = "block";
                }
                if (galleryGrid) {
                    galleryGrid.style.display = "grid";
                }
            } else {
                alert("Error generating images: " + error.message);
                // Reset UI state on error
                generateBtn.disabled = false;
                cancelBtn.style.display = "none";
                generatingPlaceholder.style.display = "none";
                if (emptyState) {
                    emptyState.style.display = "block";
                }
                if (galleryGrid) {
                    galleryGrid.style.display = "grid";
                }
            }
        } finally {
            abortController = null;
        }
    });

    // Cancel button handler - abort the request
    cancelBtn.addEventListener("click", function () {
        if (abortController) {
            abortController.abort();
        }
    });
});