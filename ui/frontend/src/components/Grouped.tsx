import { Stack, Text  } from "@mantine/core";
import { type PropsWithChildren } from "react";

export default function Grouped({ name,children }: PropsWithChildren<{name: string}>) {

  return (
    <Stack >
        <Text ta="center" style={{ width: "60%"}}>{name}</Text>
        {children}
    </Stack>
  );
}