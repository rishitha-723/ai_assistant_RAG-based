
const API_URL = window.location.origin;

const $ = id => document.getElementById(id);

const documentList = $("documentList");
const documentSelector = $("documentSelector");
const fileInput = $("fileInput");
const uploadModal = $("uploadModal");
const selectedFiles = $("selectedFiles");
const uploadBtn = $("uploadBtn");
const uploadStatus = $("uploadStatus");
const questionForm = $("questionForm");
const questionInput = $("questionInput");
const sendBtn = $("sendBtn");
const chatMessages = $("chatMessages");
const welcome = $("welcome");
const statusBadge = $("statusBadge");

let selectedPdfFiles = [];
let isAsking = false;
let isUploading = false;


async function apiRequest(path, options = {}) {
    const response = await fetch(`${API_URL}${path}`, options);

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error("The server returned an invalid response.");
    }

    if (!response.ok) {
        throw new Error(data.detail || "The request failed.");
    }

    if (data.error) {
        throw new Error(data.error);
    }

    return data;
}


function setStatus(message) {
    statusBadge.textContent = message;
}


function addTextElement(parent, tag, className, text) {
    const element = document.createElement(tag);

    if (className) element.className = className;

    element.textContent = String(text);
    parent.appendChild(element);

    return element;
}


function addMessage(role, text, sources = []) {
    welcome.classList.add("hidden");

    const message = document.createElement("article");
    message.className = `message ${role}`;

    if (role === "user") {
        message.textContent = text;
    } else {
        addTextElement(message, "div", "message-label", "StudyAI");
        addTextElement(message, "div", "answer-text", text);

        if (sources.length > 0) {
            const sourceContainer = document.createElement("div");
            sourceContainer.className = "sources";

            const uniqueSources = new Map();

            for (const source of sources) {
                const name = source.source || "Unknown document";
                const page = source.page ?? "Unknown";
                uniqueSources.set(`${name}|${page}`, { name, page });
            }

            for (const source of uniqueSources.values()) {
                addTextElement(
                    sourceContainer,
                    "span",
                    "source-pill",
                    `${source.name} · Page ${source.page}`
                );
            }

            message.appendChild(sourceContainer);
        }
    }

    chatMessages.appendChild(message);
    message.scrollIntoView({ behavior: "smooth", block: "nearest" });
}


function showError(message) {
    addMessage("assistant", message);
}


function resetChat() {
    chatMessages.replaceChildren();
    welcome.classList.remove("hidden");
    questionInput.value = "";
    questionInput.focus();
}


async function loadDocuments() {
    try {
        const data = await apiRequest("/documents");
        const previousSelection = documentSelector.value;

        documentList.replaceChildren();

        if (!data.documents || data.documents.length === 0) {
            addTextElement(
                documentList,
                "p",
                "muted",
                "No documents uploaded yet."
            );
        }

        for (const name of data.documents || []) {
            const item = document.createElement("div");
            item.className = "document-item";

            addTextElement(item, "span", "", "📄");
            addTextElement(item, "span", "document-name", name);

            const deleteButton = document.createElement("button");
            deleteButton.className = "delete-document";
            deleteButton.textContent = "×";
            deleteButton.title = `Delete ${name}`;
            deleteButton.setAttribute("aria-label", `Delete ${name}`);

            deleteButton.addEventListener("click", async () => {
                if (!confirm(`Delete "${name}" from StudyAI?`)) return;

                try {
                    setStatus("Deleting...");

                    await apiRequest(
                        `/documents/${encodeURIComponent(name)}`,
                        { method: "DELETE" }
                    );

                    await loadDocuments();
                    addMessage("assistant", `Deleted "${name}".`);
                    setStatus("Ready");
                } catch (error) {
                    showError(error.message);
                    setStatus("Error");
                }
            });

            item.appendChild(deleteButton);
            documentList.appendChild(item);
        }

        documentSelector.replaceChildren();

        const allOption = document.createElement("option");
        allOption.value = "";
        allOption.textContent = "All documents";
        documentSelector.appendChild(allOption);

        for (const name of data.documents || []) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            documentSelector.appendChild(option);
        }

        if ((data.documents || []).includes(previousSelection)) {
            documentSelector.value = previousSelection;
        }

    } catch (error) {
        console.error("Could not load documents:", error);
        setStatus("Connection error");
        documentList.replaceChildren();
        addTextElement(
            documentList,
            "p",
            "muted",
            "Could not load documents. Refresh the page."
        );
    }
}


function openUploadModal() {
    uploadModal.classList.remove("hidden");
}


function closeUploadModal() {
    if (!isUploading) {
        uploadModal.classList.add("hidden");
    }
}


function updateSelectedFiles() {
    selectedFiles.replaceChildren();

    for (const file of selectedPdfFiles) {
        addTextElement(selectedFiles, "div", "", `📄 ${file.name}`);
    }

    uploadBtn.disabled = selectedPdfFiles.length === 0 || isUploading;

    if (selectedPdfFiles.length > 0) {
        uploadStatus.textContent =
            `${selectedPdfFiles.length} PDF file(s) selected.`;
    } else {
        uploadStatus.textContent = "";
    }
}


fileInput.addEventListener("change", () => {
    const files = Array.from(fileInput.files || []);

    const validFiles = files.filter(file =>
        file.name.toLowerCase().endsWith(".pdf") &&
        (file.type === "application/pdf" || file.type === "")
    );

    const invalidCount = files.length - validFiles.length;
    const existing = new Set(selectedPdfFiles.map(file => file.name));

    for (const file of validFiles) {
        if (!existing.has(file.name)) {
            selectedPdfFiles.push(file);
            existing.add(file.name);
        }
    }

    updateSelectedFiles();

    if (invalidCount > 0) {
        uploadStatus.textContent = "Only PDF files are accepted.";
    }

    fileInput.value = "";
});


uploadBtn.addEventListener("click", async () => {
    if (selectedPdfFiles.length === 0 || isUploading) return;

    isUploading = true;
    uploadBtn.disabled = true;
    setStatus("Uploading...");

    let successful = 0;
    const failures = [];

    try {
        for (let i = 0; i < selectedPdfFiles.length; i++) {
            const file = selectedPdfFiles[i];

            uploadStatus.textContent =
                `Uploading ${i + 1} of ${selectedPdfFiles.length}: ${file.name}`;

            const formData = new FormData();
            formData.append("file", file);

            try {
                await apiRequest("/upload", {
                    method: "POST",
                    body: formData
                });

                successful++;
            } catch (error) {
                failures.push(`${file.name}: ${error.message}`);
            }
        }

        await loadDocuments();

        if (successful > 0) {
            addMessage(
                "assistant",
                `Successfully added ${successful} PDF document(s). You can now ask questions about them.`
            );
        }

        if (failures.length > 0) {
            showError("Some files could not be uploaded:\n" + failures.join("\n"));
        }

        selectedPdfFiles = [];
        updateSelectedFiles();

        if (failures.length === 0) {
            uploadModal.classList.add("hidden");
        }

        setStatus(failures.length > 0 ? "Check uploads" : "Ready");

    } catch (error) {
        showError(error.message);
        setStatus("Error");
    } finally {
        isUploading = false;
        uploadBtn.disabled = selectedPdfFiles.length === 0;
    }
});


questionForm.addEventListener("submit", async event => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question || isAsking) return;

    isAsking = true;
    sendBtn.disabled = true;
    questionInput.disabled = true;

    addMessage("user", question);
    questionInput.value = "";
    setStatus("Thinking...");

    const loading = document.createElement("article");
    loading.className = "message assistant";
    loading.textContent = "Finding an answer...";
    chatMessages.appendChild(loading);

    try {
        const formData = new FormData();
        formData.append("question", question);
        formData.append("document_name", documentSelector.value);

        const data = await apiRequest("/ask", {
            method: "POST",
            body: formData
        });

        loading.remove();

        addMessage(
            "assistant",
            data.answer || "I couldn't generate an answer.",
            data.sources || []
        );

        setStatus("Ready");

    } catch (error) {
        loading.remove();
        showError(error.message);
        setStatus("Error");
    } finally {
        isAsking = false;
        sendBtn.disabled = false;
        questionInput.disabled = false;
        questionInput.focus();
    }
});


$("addDocumentsBtn").addEventListener("click", openUploadModal);
$("closeModalBtn").addEventListener("click", closeUploadModal);
$("newChatBtn").addEventListener("click", resetChat);


$("clearDocumentsBtn").addEventListener("click", async () => {
    if (!confirm("Clear all uploaded documents from StudyAI?")) return;

    try {
        setStatus("Clearing...");

        await apiRequest("/clear", { method: "DELETE" });
        await loadDocuments();

        addMessage(
            "assistant",
            "All documents have been cleared. Upload a PDF to start again."
        );

        setStatus("Ready");
    } catch (error) {
        showError(error.message);
        setStatus("Error");
    }
});


$("themeBtn").addEventListener("click", () => {
    document.body.classList.toggle("dark");
});


document.querySelectorAll(".suggestion").forEach(button => {
    button.addEventListener("click", () => {
        questionInput.value = button.dataset.question || "";
        questionInput.focus();
    });
});


uploadModal.addEventListener("click", event => {
    if (event.target === uploadModal) closeUploadModal();
});


loadDocuments();
