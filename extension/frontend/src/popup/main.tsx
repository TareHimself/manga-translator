import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Import styles of packages that you've installed.
// All packages except `@mantine/hooks` require styles imports
import "@mantine/core/styles.css";
import "./index.css";
import App from "./App.tsx";
import { MantineProvider } from "@mantine/core";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MantineProvider defaultColorScheme="dark">
      <App />
    </MantineProvider>
  </StrictMode>
);
