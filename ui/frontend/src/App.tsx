// core styles are required for all packages
import "@mantine/core/styles.css";
import { useEffect } from "react";
import ImageConverter from "./components/ImageConverter";
import ImageSettings from "./components/ImageSettings";
import { useAppStore } from "./useAppStore";
import { createTheme, MantineProvider } from "@mantine/core";

const theme = createTheme({
  /** Your theme override here */
});

function App() {
  useEffect(() => {
    useAppStore.getState().getServerInfo().catch(console.error);
  }, []);

  return (
    <MantineProvider theme={theme} forceColorScheme="dark">
      <div className="app">
        <ImageSettings />
        <ImageConverter />
      </div>
    </MantineProvider>
  );
}

export default App;
