import { useEffect } from "react";
import ImageConverter from "./components/ImageConverter";
import ImageSettings from "./components/ImageSettings";
import { useAppDispatch } from "./redux/hooks";
import { getServerInfo } from "./redux/slices/app";

function App() {
  const dispatch = useAppDispatch();

  useEffect(() => {
    dispatch(getServerInfo());
  }, [dispatch]);
  return (
    <div className="app">
      <ImageSettings />
      <ImageConverter />
    </div>
  );
}

export default App;
