import "./App.css";
import { NumberInput, Stack, TextInput } from "@mantine/core";
import { useExtensionStore } from "./hooks/useExtensionStore";

function LoadedApp() {
  const serverAddress = useExtensionStore((s) => s.serverAddress);
  const batchSize = useExtensionStore((s) => s.batchSize);
  const batchTimeout = useExtensionStore((s) => s.batchTimeout);
  const setServerAddress = useExtensionStore((s) => s.setServerAddress);
  const setBatchSize = useExtensionStore((s) => s.setBatchSize);
  const setBatchTimeout = useExtensionStore((s) => s.setBatchTimeout);
  return (
    <Stack>
      <TextInput
        value={serverAddress}
        placeholder="https://someserver.com/api/v1"
        label="Server API Addesss"
        onChange={(e) => {
          setServerAddress(e.target.value);
        }}
      />
      <NumberInput
        label="Batch Size"
        min={1}
        allowDecimal={false}
        value={batchSize}
        description="Maximum size of a translation batch"
        onChange={(e) => {
          if (typeof e === "string") {
            setBatchSize(parseInt(e));
          } else {
            setBatchSize(e);
          }
        }}
      />
      <NumberInput
        label="Batch Timeout"
        min={0}
        value={batchTimeout}
        description="Time in milliseconds before attempting to translate an incomplete batch"
        onChange={(e) => {
          if (typeof e === "string") {
            setBatchTimeout(parseInt(e));
          } else {
            setBatchTimeout(e);
          }
        }}
      />
    </Stack>
  );
}
function App() {
  const loading = useExtensionStore((s) => s.loading);

  if (loading) {
    return <div className="loader" />;
  }

  return <LoadedApp />;
}

export default App;
