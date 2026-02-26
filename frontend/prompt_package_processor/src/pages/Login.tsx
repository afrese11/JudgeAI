import { useState } from "react";
import { useNavigate } from "react-router-dom";

const STORAGE_KEY = "judgeai_passcode";

export default function Login() {
  const [passcode, setPasscode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!passcode.trim()) {
      setError("Enter the access code.");
      return;
    }

    // Store locally so they only type it once
    localStorage.setItem(STORAGE_KEY, passcode.trim());
    navigate("/", { replace: true });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-6">
      <div className="w-full max-w-md rounded-xl border p-6 shadow-sm bg-card">
        <h1 className="text-2xl font-bold mb-2">JudgeAI</h1>
        <p className="text-muted-foreground mb-6">
          Enter the access code to continue.
        </p>

        <form onSubmit={onSubmit} className="space-y-4">
          <input
            value={passcode}
            onChange={(e) => setPasscode(e.target.value)}
            placeholder="Access code"
            className="w-full h-12 px-3 rounded-md border bg-background"
            autoFocus
          />

          {error ? (
            <div className="text-sm text-red-500">{error}</div>
          ) : null}

          <button
            type="submit"
            className="w-full h-12 rounded-md bg-primary text-primary-foreground font-medium"
          >
            Continue
          </button>
        </form>
      </div>
    </div>
  );
}