// src/pages/Index.tsx
import { useMemo, useState } from 'react';
import { FileDropZone } from '@/components/FileDropZone';
import { OutputDisplay } from '@/components/OutputDisplay';
import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { supabase } from "@/lib/supabase";

const Index = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // If VITE_API_BASE_URL is unset, use relative /api (works if you proxy locally,
  // or if you serve frontend+backend behind the same domain in prod).
  // Vite fix
  const apiBase = useMemo(() => {
    const envBase = import.meta.env.VITE_API_BASE_URL as string | undefined;
    return (envBase && envBase.trim()) ? envBase.trim().replace(/\/+$/, '') : '';
  }, []);

  const handleProcess = async () => {
    setOutput(`DEBUG: apiBase=${apiBase}`);
    if (files.length === 0) return;

    setIsLoading(true);
    setOutput("");

    try {
      const passcode = localStorage.getItem("judgeai_passcode") || "";
      if (!passcode.trim()) throw new Error("Missing access code. Please log in again.");

      // 1) Ask backend for signed upload URLs (one per file)
      const initUrl = `${apiBase}/api/uploads/init`;
      const initRes = await fetch(initUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-JudgeAI-Passcode": passcode,
        },
        body: JSON.stringify({
          files: files.map((f) => ({
            filename: f.name,
            content_type: f.type || "application/pdf",
          })),
        }),
      });

      const initJson = await initRes.json().catch(() => ({}));
      if (!initRes.ok || initJson?.error) {
        throw new Error(initJson?.error || `Init upload failed (${initRes.status})`);
      }

      const bucket: string = initJson.bucket || "judgeai-pdfs";
      const uploads: Array<{ path: string; signed_url: string; token: string }> = initJson.uploads;

      if (!uploads || uploads.length !== files.length) {
        throw new Error("Upload init mismatch (did not receive signed URLs for all files).");
      }

      // 2) Upload each file directly to Supabase using the signed upload token
      for (let i = 0; i < files.length; i++) {
        const f = files[i];
        const u = uploads[i];

        const { error } = await supabase.storage
          .from(bucket)
          .uploadToSignedUrl(u.path, u.token, f, {
            contentType: "application/pdf",
          });

        if (error) throw new Error(`Supabase upload failed for ${f.name}: ${error.message}`);
      }

      // 3) Ask backend to process from storage paths (tiny JSON request)
      const judgeUrl = `${apiBase}/api/judge-from-storage`;
      const judgeRes = await fetch(judgeUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-JudgeAI-Passcode": passcode,
        },
        body: JSON.stringify({
          paths: uploads.map((u) => u.path),
          redact: false,
          cleanup: true,
        }),
      });

      const data: {
        raw?: string;
        case_summary?: string | null;
        case_decision?: string | null;
        error?: string;
      } = await judgeRes.json();

      if (!judgeRes.ok || data.error) {
        throw new Error(data.error || `Judge failed (${judgeRes.status})`);
      }

      const pretty =
        data.case_summary && data.case_decision
          ? [
              "===CASE SUMMARY===",
             data.case_summary,
             "",
              "===CASE DECISION===",
              data.case_decision,
            ].join("\n")
          : data.raw || "No output returned.";

      setOutput(pretty);
    } catch (err: any) {
      setOutput(`Error: ${err?.message ?? String(err)}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container max-w-3xl py-12 px-4">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-foreground mb-2">JudgeAI</h1>
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
