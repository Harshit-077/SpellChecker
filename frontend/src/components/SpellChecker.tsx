import { useState } from "react";
import { Button } from "./ui/button";
import { Wand2, Copy, RefreshCw, ArrowRight, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";

const SAMPLE_TEXTS = [
  { input: "रॉलिंग", expected: "रॉलिंग" },
  { input: "ओल्मपियन", expected: "ओलम्पियन" },
  { input: "पुश्तैनी", expected: "पुश्तैनी" },
  { input: "तम्बोली", expected: "तम्बोली" },
  { input: "कनका", expected: "कनका" },
];

const SpellChecker = () => {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasChecked, setHasChecked] = useState(false);

  const handleCheckSpelling = async () => {
    if (!inputText.trim()) {
      toast.error("कृपया कुछ टेक्स्ट दर्ज करें");
      return;
    }

    setIsProcessing(true);
    setHasChecked(false);

    // Simulate processing delay for demo
    // await new Promise((resolve) => setTimeout(resolve, 1500));

    // // Demo: For now, show the input as output (real implementation would call backend)
    // // In production, this would send to a Python backend with the seq2seq model
    // setOutputText(inputText);



    try {
      const response = await fetch("http://127.0.0.1:8000/api/spell-check", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: inputText,
        }),
      });

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();

      setOutputText(data.corrected);
      toast.success("वर्तनी जाँच पूर्ण!");
    } catch (err) {
      toast.error("सर्वर से कनेक्ट नहीं हो पाया");
    } finally {
      setIsProcessing(false);
      setHasChecked(true);
  }





















    setIsProcessing(false);
    setHasChecked(true);
    toast.success("वर्तनी जाँच पूर्ण!");
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(outputText);
    toast.success("कॉपी किया गया!");
  };

  const handleClear = () => {
    setInputText("");
    setOutputText("");
    setHasChecked(false);
  };

  const handleSampleClick = (sample: { input: string; expected: string }) => {
    setInputText(sample.input);
    setOutputText("");
    setHasChecked(false);
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4 md:px-8 py-8">
      {/* Sample Texts */}
      <div className="mb-8 animate-fade-in">
        <p className="text-sm font-medium text-muted-foreground mb-3">
          उदाहरण टेक्स्ट आज़माएं:
        </p>
        <div className="flex flex-wrap gap-2">
          {SAMPLE_TEXTS.map((sample, index) => (
            <button
              key={index}
              onClick={() => handleSampleClick(sample)}
              className="px-4 py-2 bg-secondary hover:bg-secondary/80 rounded-lg text-sm font-hindi text-secondary-foreground transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
            >
              {sample.input}
            </button>
          ))}
        </div>
      </div>

      {/* Main Editor Section */}
      <div className="grid md:grid-cols-2 gap-6 animate-fade-in" style={{ animationDelay: "0.1s" }}>
        {/* Input Panel */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold text-foreground flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-primary"></span>
              इनपुट टेक्स्ट
            </label>
            <span className="text-xs text-muted-foreground">
              {inputText.length} अक्षर
            </span>
          </div>
          <div className="relative">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="यहाँ हिंदी में टाइप करें..."
              className="w-full h-64 p-4 bg-card border-2 border-border rounded-2xl resize-none font-hindi text-lg leading-relaxed placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary focus:ring-4 focus:ring-primary/10 transition-all duration-300 shadow-card"
              dir="auto"
            />
          </div>
        </div>

        {/* Output Panel */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold text-foreground flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-accent"></span>
              सुधारित टेक्स्ट
            </label>
            {hasChecked && (
              <span className="flex items-center gap-1 text-xs text-accent">
                <CheckCircle2 className="w-3 h-3" />
                जाँचा गया
              </span>
            )}
          </div>
          <div className="relative">
            <div className="w-full h-64 p-4 bg-card border-2 border-border rounded-2xl font-hindi text-lg leading-relaxed overflow-auto shadow-card">
              {isProcessing ? (
                <div className="flex flex-col items-center justify-center h-full gap-3">
                  <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                  <p className="text-sm text-muted-foreground">जाँच हो रही है...</p>
                </div>
              ) : outputText ? (
                <p className="text-foreground">{outputText}</p>
              ) : (
                <p className="text-muted-foreground/50">
                  सुधारित टेक्स्ट यहाँ दिखाई देगा...
                </p>
              )}
            </div>
            {outputText && !isProcessing && (
              <button
                onClick={handleCopy}
                className="absolute top-4 right-4 p-2 bg-secondary hover:bg-secondary/80 rounded-lg transition-all duration-200"
                title="कॉपी करें"
              >
                <Copy className="w-4 h-4 text-secondary-foreground" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mt-8 animate-fade-in" style={{ animationDelay: "0.2s" }}>
        <Button
          variant="hero"
          size="xl"
          onClick={handleCheckSpelling}
          disabled={isProcessing || !inputText.trim()}
          className="w-full sm:w-auto"
        >
          {isProcessing ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin" />
              जाँच हो रही है...
            </>
          ) : (
            <>
              <Wand2 className="w-5 h-5" />
              वर्तनी जाँचें
              <ArrowRight className="w-5 h-5" />
            </>
          )}
        </Button>
        <Button
          variant="outline"
          size="lg"
          onClick={handleClear}
          disabled={isProcessing}
          className="w-full sm:w-auto"
        >
          <RefreshCw className="w-4 h-4" />
          साफ़ करें
        </Button>
      </div>
    </div>
  );
};

export default SpellChecker;
