import { Brain, Cpu, Database, Zap, FileCode, Layers } from "lucide-react";

const ModelInfo = () => {
  const specs = [
    {
      icon: Brain,
      label: "Architecture",
      value: "Seq2Seq LSTM",
    },
    {
      icon: Layers,
      label: "Layers",
      value: "2 Encoder + 2 Decoder",
    },
    {
      icon: Database,
      label: "Vocabulary",
      value: "91 Characters",
    },
    {
      icon: Cpu,
      label: "Embed Size",
      value: "192 dims",
    },
    {
      icon: Zap,
      label: "Hidden Size",
      value: "256 units",
    },
    {
      icon: FileCode,
      label: "Training Data",
      value: "100K sentences",
    },
  ];

  return (
    <div className="w-full max-w-6xl mx-auto px-4 md:px-8 py-12">
      <div className="text-center mb-8 animate-fade-in">
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-2">
          Model Information
        </h2>
        <p className="text-muted-foreground font-hindi">
          seq2seq_best.pt - Character-Level Hindi Spell Correction
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 animate-fade-in" style={{ animationDelay: "0.1s" }}>
        {specs.map((spec, index) => (
          <div
            key={index}
            className="flex flex-col items-center p-4 bg-card rounded-2xl border border-border shadow-card hover:shadow-glow transition-all duration-300 hover:scale-[1.02] group"
            style={{ animationDelay: `${0.1 + index * 0.05}s` }}
          >
            <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center mb-3 group-hover:bg-primary/10 transition-colors duration-300">
              <spec.icon className="w-6 h-6 text-primary" />
            </div>
            <span className="text-xs text-muted-foreground text-center mb-1">
              {spec.label}
            </span>
            <span className="text-sm font-semibold text-foreground text-center">
              {spec.value}
            </span>
          </div>
        ))}
      </div>

      {/* How it works */}
      <div className="mt-12 p-6 md:p-8 bg-card rounded-2xl border border-border shadow-card animate-fade-in" style={{ animationDelay: "0.3s" }}>
        <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg gradient-saffron flex items-center justify-center">
            <Zap className="w-4 h-4 text-primary-foreground" />
          </div>
          How It Works
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <div className="text-sm font-medium text-primary">1. Input Processing</div>
            <p className="text-sm text-muted-foreground">
              Hindi text is tokenized at the character level. Each character is mapped to a unique ID using a vocabulary of 91 characters including special tokens.
            </p>
          </div>
          <div className="space-y-2">
            <div className="text-sm font-medium text-primary">2. Encoding</div>
            <p className="text-sm text-muted-foreground">
              The encoder LSTM processes the input sequence, learning contextual representations of potentially misspelled characters.
            </p>
          </div>
          <div className="space-y-2">
            <div className="text-sm font-medium text-primary">3. Decoding</div>
            <p className="text-sm text-muted-foreground">
              The decoder generates corrected text character-by-character, using learned patterns to fix typos like deletions, replacements, and transpositions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInfo;
