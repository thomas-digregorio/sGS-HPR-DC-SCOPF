import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const title = "HPR Reproduction Control Room";
const description =
  "A stage-gated research dashboard for reproducing the GPU-based sGS-HPR DCOPF paper on an NVIDIA DGX Spark.";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const metadataBase = new URL(`${protocol}://${host}`);
  const imageUrl = new URL("/og-stage7.png", metadataBase).toString();

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
      images: [{ url: imageUrl, width: 1536, height: 1024, alt: title }],
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
