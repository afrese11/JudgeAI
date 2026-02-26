import { Navigate, Outlet, useLocation } from "react-router-dom";

const STORAGE_KEY = "judgeai_passcode";

export default function RequirePasscode() {
  const location = useLocation();
  const code = localStorage.getItem(STORAGE_KEY);

  if (!code || !code.trim()) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return <Outlet />;
}