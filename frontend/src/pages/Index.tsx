import { useState } from "react";
import Header from "@/components/Header";
import SpellChecker from "@/components/SpellChecker";
import ModelInfo from "@/components/ModelInfo";
import BackendGuide from "@/components/BackendGuide";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Wand2, Code } from "lucide-react";

const Index = () => {
  const [activeTab, setActiveTab] = useState<"demo" | "guide">("demo");

  return (
    <div className="min-h-screen gradient-hero flex flex-col">
      <Header />
      
      {/* Hero Section */}
      <section className="py-8 md:py-12 px-4">
        <div className="max-w-3xl mx-auto text-center animate-fade-in">
          <h1 className="text-3xl md:text-5xl font-bold text-foreground mb-4 leading-tight">
            Intelligent{" "}
            <span className="text-gradient-saffron">Spell Checker</span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground font-hindi leading-relaxed">
            Character-level Seq2Seq model trained on 100,000+ Hindi sentences.
            <br />
            <span className="text-foreground font-medium">80% accuracy</span> on typo correction.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center justify-center gap-2 mt-8">
          {/* <Button
            variant={activeTab === "demo" ? "default" : "secondary"}
            size="lg"
            onClick={() => setActiveTab("demo")}
            className="gap-2"
          > */}
            {/* <Wand2 className="w-4 h-4" /> */}
            {/* Try Demo */}
          {/* </Button> */}
          {/* <Button
            variant={activeTab === "guide" ? "default" : "secondary"}
            size="lg"
            onClick={() => setActiveTab("guide")}
            className="gap-2"
          >
            <Code className="w-4 h-4" />
            Backend Guide
          </Button> */}
        </div>
      </section>

      {/* Main Content */}
      <main className="flex-grow">
        {activeTab === "demo" ? (
          <>
            <SpellChecker />
            <ModelInfo />
          </>
        ) : (
          <BackendGuide />
        )}
      </main>

      <Footer />
    </div>
  );
};

export default Index;
