import { useState } from "react";

export default function SpellChecker() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");

  const checkSpelling = async () => {
    const res = await fetch("http://localhost:8000/api/spell-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: inputText }),
    });

    const data = await res.json();
    setOutputText(data.corrected);
  };

  return (
    <div style={{ padding: "20px" }}>
      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Enter Hindi text"
      />

      <br />

      <button onClick={checkSpelling}>Check</button>

      <h3>Corrected:</h3>
      <p>{outputText}</p>
    </div>
  );
}
