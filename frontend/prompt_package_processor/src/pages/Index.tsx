import { useState } from 'react';
import { FileDropZone } from '@/components/FileDropZone';
import { OutputDisplay } from '@/components/OutputDisplay';
import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';

// HARDCODED PROMPT - Edit this to change the processing behavior
const SYSTEM_PROMPT = `Analyze the provided files and generate a summary of their contents. 
Include: file names, types, and key information found in each file.`;

const Index = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleProcess = async () => {
    if (files.length === 0) return;

    setIsLoading(true);
    setOutput('');

    // Simulate processing - Replace this with actual API call
    // The SYSTEM_PROMPT above should be sent along with file contents
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Mock output - Replace with actual API response
    const mockOutput = `Processed ${files.length} file(s) with prompt:\n\n"${SYSTEM_PROMPT}"\n\nFiles analyzed:\n${files.map((f, i) => `${i + 1}. ${f.name} (${(f.size / 1024).toFixed(1)} KB)`).join('\n')}\n\n[Connect to your backend/API to get real processing results]`;

    setOutput(mockOutput);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container max-w-3xl py-12 px-4">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            JudgeAI
          </h1>
          <p className="text-muted-foreground">
            Upload legal documents for AI analysis.
          </p>
        </div>

        <div className="space-y-6">
          <FileDropZone files={files} onFilesChange={setFiles} />

          <Button
            onClick={handleProcess}
            disabled={files.length === 0 || isLoading}
            className="w-full h-12 text-base font-medium glow-button"
            size="lg"
          >
            <Sparkles className="w-5 h-5 mr-2" />
            {isLoading ? 'Processing...' : 'Go'}
          </Button>

          <OutputDisplay output={output} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
};

export default Index;
