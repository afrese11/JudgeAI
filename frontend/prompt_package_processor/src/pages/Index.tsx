import { useEffect, useMemo, useState } from 'react';
import { FileDropZone } from '@/components/FileDropZone';
import { OutputDisplay } from '@/components/OutputDisplay';
import { Sparkles, ShieldCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { submitJudgeCase, type JudgeCaseResponse } from '@/lib/api';
import { isSupabaseConfigured, supabase } from '@/lib/supabase';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import type { Session } from '@supabase/supabase-js';

const ALLOWED_SIGNUP_EMAILS = new Set([
  'david.schwartz@law.northwestern.edu',
  'andrewfrese2027@u.northwestern.edu',
  'nikola.datzov@und.edu',
  'muhammadrozaidi2026@u.northwestern.edu',
  'finnmcm@u.northwestern.edu',
]);

const Index = () => {
  const [addendumFiles, setAddendumFiles] = useState<File[]>([]);
  const [relatedFiles, setRelatedFiles] = useState<File[]>([]);
  const [result, setResult] = useState<JudgeCaseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [redact, setRedact] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [session, setSession] = useState<Session | null>(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [authNotice, setAuthNotice] = useState<string | null>(null);
  const [isAuthSubmitting, setIsAuthSubmitting] = useState(false);

  const apiBase = useMemo(() => {
    const envBase = import.meta.env.VITE_API_BASE_URL as string | undefined;
    if (envBase && envBase.trim()) {
      return envBase.trim().replace(/\/+$/, '');
    }
    // Local dev fallback when Vite proxy is unavailable.
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://127.0.0.1:8000';
    }
    return '';
  }, []);

  const handleValidationError = (message: string) => {
    setError(message);
  };

  useEffect(() => {
    if (!isSupabaseConfigured || !supabase) {
      return;
    }
    void supabase.auth.getSession().then(({ data }) => {
      setSession(data.session ?? null);
    });
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
    });
    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const validateAuthInputs = (): boolean => {
    if (!email.trim() || !password.trim()) {
      setError('Enter both email and password.');
      return false;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return false;
    }
    return true;
  };

  const handleEmailSignIn = async () => {
    if (!isSupabaseConfigured || !supabase) {
      setError('Supabase is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
      return;
    }
    if (!validateAuthInputs()) {
      return;
    }
    setIsAuthSubmitting(true);
    setError(null);
    setAuthNotice(null);
    const { error: signInError } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });
    if (signInError) {
      setError(signInError.message);
    }
    setIsAuthSubmitting(false);
  };

  const handleEmailSignUp = async () => {
    if (!isSupabaseConfigured || !supabase) {
      setError('Supabase is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
      return;
    }
    if (!validateAuthInputs()) {
      return;
    }
    const normalizedEmail = email.trim().toLowerCase();
    if (!ALLOWED_SIGNUP_EMAILS.has(normalizedEmail)) {
      setError('This email is not approved for account creation.');
      return;
    }
    setIsAuthSubmitting(true);
    setError(null);
    setAuthNotice(null);
    const { data, error: signUpError } = await supabase.auth.signUp({
      email: normalizedEmail,
      password,
    });
    if (signUpError) {
      setError(signUpError.message);
    } else if (!data.session) {
      setAuthNotice('Check your email to confirm your account, then sign in.');
    } else {
      setAuthNotice('Account created and signed in.');
    }
    setIsAuthSubmitting(false);
  };

  const handleSignOut = async () => {
    if (!supabase) {
      return;
    }
    const { error: signOutError } = await supabase.auth.signOut();
    if (signOutError) {
      setError(signOutError.message);
    }
  };

  const handleAnalyze = async () => {
    if (addendumFiles.length === 0) {
      setError('Upload one addendum PDF before running analysis.');
      return;
    }
    if (!session?.access_token || !supabase) {
      setError('Sign in before analyzing briefs.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const {
        data: { session: latestSession },
        error: sessionError,
      } = await supabase.auth.getSession();
      if (sessionError) {
        throw new Error(sessionError.message);
      }
      if (!latestSession?.access_token) {
        throw new Error('Your session expired. Sign in again.');
      }

      const response = await submitJudgeCase({
        caseAddendumFile: addendumFiles[0],
        relatedFiles,
        redact,
        apiBase,
        accessToken: latestSession.access_token,
      });
      setResult(response);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const isAuthenticated = Boolean(session?.user && session.access_token);

  return (
    <div className="min-h-screen bg-background">
      <div className="container max-w-3xl py-12 px-4">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-foreground mb-2">JudgeAI</h1>
          <p className="text-muted-foreground">
            Upload one case addendum and optional related briefs for a concise structured analysis.
          </p>
        </div>

        <div className="space-y-6">
          {error ? (
            <Alert variant="destructive">
              <AlertTitle>Something went wrong</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <Alert>
            <ShieldCheck className="h-4 w-4" />
            <AlertTitle>Authentication</AlertTitle>
            <AlertDescription>
              {isSupabaseConfigured
                ? session?.user?.email
                  ? `Signed in as ${session.user.email}.`
                  : 'Sign in with email and password to call secured backend endpoints.'
                : 'Supabase auth is not configured in this frontend environment.'}
            </AlertDescription>
          </Alert>

          {authNotice ? (
            <Alert>
              <AlertTitle>Authentication update</AlertTitle>
              <AlertDescription>{authNotice}</AlertDescription>
            </Alert>
          ) : null}

          <div className="space-y-3">
            {session?.user ? (
              <Button
                type="button"
                variant="outline"
                onClick={handleSignOut}
                disabled={isLoading || isAuthSubmitting}
              >
                Sign out
              </Button>
            ) : (
              <>
                <Input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  disabled={isLoading || isAuthSubmitting}
                  autoComplete="email"
                />
                <Input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  disabled={isLoading || isAuthSubmitting}
                  autoComplete="current-password"
                />
                <div className="flex gap-3">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleEmailSignIn}
                    disabled={isLoading || isAuthSubmitting}
                  >
                    {isAuthSubmitting ? 'Signing in...' : 'Sign in'}
                  </Button>
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={handleEmailSignUp}
                    disabled={isLoading || isAuthSubmitting}
                  >
                    {isAuthSubmitting ? 'Signing up...' : 'Sign up'}
                  </Button>
                </div>
              </>
            )}
          </div>

          {!isAuthenticated ? (
            <Alert>
              <AlertTitle>Sign in required</AlertTitle>
              <AlertDescription>
                {isSupabaseConfigured
                  ? 'Authenticate with email and password to continue to case analysis.'
                  : 'Authentication is unavailable until Supabase environment variables are set.'}
              </AlertDescription>
            </Alert>
          ) : (
            <>
              <FileDropZone
                files={addendumFiles}
                onFilesChange={setAddendumFiles}
                multiple={false}
                title="Case Addendum (required)"
                description="Upload exactly one PDF addendum document."
                onValidationError={handleValidationError}
                disabled={isLoading}
              />
              <FileDropZone
                files={relatedFiles}
                onFilesChange={setRelatedFiles}
                multiple
                title="Related Files (optional)"
                description="Upload any additional PDF briefs or related documents."
                onValidationError={handleValidationError}
                disabled={isLoading}
              />

              <div className="flex items-center justify-between rounded-lg border bg-card px-4 py-3">
                <label htmlFor="redact-toggle" className="text-sm font-medium text-foreground">
                  Redact sensitive details before analysis
                </label>
                <input
                  id="redact-toggle"
                  type="checkbox"
                  checked={redact}
                  disabled={isLoading}
                  onChange={(event) => setRedact(event.target.checked)}
                  className="h-4 w-4 accent-primary"
                />
              </div>

              <Button
                onClick={handleAnalyze}
                disabled={addendumFiles.length === 0 || isLoading}
                className="w-full h-12 text-base font-medium glow-button"
                size="lg"
              >
                <Sparkles className="w-5 h-5 mr-2" />
                {isLoading ? 'Analyzing documents...' : 'Analyze Case'}
              </Button>

              <OutputDisplay result={result} error={error} isLoading={isLoading} />
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;
