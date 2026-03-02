async function analyzeReview() {
  const input = document.getElementById("reviewInput");
  const msg = document.getElementById("message");
  const btn = document.getElementById("analyzeBtn");
  const text = input.value.trim();
  msg.classList.add("hidden");
  if (!text) {
    msg.textContent = "Please enter a review.";
    msg.classList.remove("hidden");
    return;
  }
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = "Analyzing...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review: text }),
    });
    if (!res.ok) {
      throw new Error("Request failed");
    }
    const data = await res.json();
    document.getElementById("theme").innerText = data.theme;
    const s = document.getElementById("sentiment");
    s.innerText = data.sentiment;
    s.classList.remove("pos","neg","neu");
    if (data.sentiment === "Positive") s.classList.add("pos");
    else if (data.sentiment === "Negative") s.classList.add("neg");
    else s.classList.add("neu");
    document.getElementById("result").classList.remove("hidden");
  } catch (e) {
    msg.textContent = "Service temporarily unavailable. Try again.";
    msg.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

function clearReview() {
  document.getElementById("reviewInput").value = "";
  document.getElementById("theme").innerText = "";
  document.getElementById("sentiment").innerText = "";
  document.getElementById("result").classList.add("hidden");
  const msg = document.getElementById("message");
  msg.textContent = "";
  msg.classList.add("hidden");
}

// Show build version so cache issues are obvious
async function showVersion() {
  try {
    const res = await fetch("/version");
    const data = await res.json();
    document.getElementById("versionTag").innerText = data.version;
  } catch (e) {
    document.getElementById("versionTag").innerText = "dev";
  }
}
showVersion();
