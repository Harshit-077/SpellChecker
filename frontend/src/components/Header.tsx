import { Sparkles, Info } from "lucide-react";

const Header = () => {
  return (
    <header className="w-full py-6 px-4 md:px-8">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl gradient-saffron flex items-center justify-center shadow-glow">
            <span className="text-2xl font-bold text-primary-foreground font-hindi">अ</span>
          </div>
          <div>
            <h1 className="text-xl md:text-2xl font-bold text-foreground">
              Hindi Spell Checker
            </h1>
            <p className="text-sm text-muted-foreground font-hindi">
              हिंदी वर्तनी जाँचक
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="hidden md:flex items-center gap-2 px-4 py-2 bg-secondary rounded-lg">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-secondary-foreground">
              Seq2Seq Model
            </span>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 bg-accent/10 rounded-lg">
            <Info className="w-4 h-4 text-accent" />
            <span className="text-sm font-medium text-accent">94.38% Accuracy</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
