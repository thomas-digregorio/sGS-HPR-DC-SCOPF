import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const title = "HPR Reproduction Control Room";
const description =
  "Stage 9 scientific report and evidence dashboard for the structural reproduction of the GPU-based sGS-HPR DCOPF paper on NVIDIA DGX Spark.";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const metadataBase = new URL(`${protocol}://${host}`);
  const imageUrl = new URL("/og-stage9.png", metadataBase).toString();

  return {
    metadataBase,
    title: {
      default: title,
      template: "%s | HPR Reproduction",
    },
    description,
    openGraph: {
      title,
      description,
      type: "website",
      images: [{ url: imageUrl, width: 1672, height: 941, alt: title }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [imageUrl],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
