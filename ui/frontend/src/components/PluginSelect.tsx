import { type ComboboxData, type ComboboxItem, Select } from "@mantine/core";
import { useMemo } from "react";
import type { IPlugin } from "../types";

export type PluginSelectProps = {
  name: string;
  selected: number
  items: IPlugin[]
  onChange: (selected: number) => void
}

export default function PluginSelect(props: PluginSelectProps) {
  const data = useMemo<ComboboxData>(
    () =>
      props.items.map<ComboboxItem>((c,idx) => ({
        label: c.name,
        value: idx.toString(),
      })),
    [props]
  );
  
  return (
    <Select
      label={props.name}
      description={(props.selected != -1 && props.selected < props.items.length) ? props.items[props.selected].description : undefined}
      data={data}
      value={props.selected.toString()}
      onChange={(value) => props.onChange(parseInt(value ?? "0"))}
      style={{ width: '60%' }}
    />
  );
}
