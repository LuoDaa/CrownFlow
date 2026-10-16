import React from "react";
import ReactDOM from "react-dom/client";
import "./domain/v1TestOverrides";
import "./domain/v1DemoOverrides";
import App from "./App";
ReactDOM.createRoot(document.getElementById("root")).render(<React.StrictMode><App/></React.StrictMode>);
