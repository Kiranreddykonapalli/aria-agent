"""
Email Agent: sends the Aria analytics report and charts via email.

Responsibilities:
  - Compose a professional HTML email from the report text and figure paths
  - Embed chart PNGs as inline images (Content-ID references)
  - Attach the full markdown report as a .md file
  - Send via SMTP using credentials from environment variables
  - Return a result dict: success, message, recipient

Uses only Python standard library (smtplib, email.*) — no extra dependencies.
SMTP settings are read from: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD.
"""

from __future__ import annotations

import os
import re
import smtplib
import ssl
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


class EmailAgent:
    def __init__(self) -> None:
        self.smtp_host     = os.getenv("SMTP_HOST", "")
        self.smtp_port     = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user     = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")

    def run(
        self,
        recipient_email: str,
        report_text: str,
        figure_paths: list[str],
        question: str,
    ) -> dict:
        """
        Send the analytics report and charts to recipient_email.

        Args:
            recipient_email: Destination email address.
            report_text:     Full markdown report from ReportWriter.
            figure_paths:    List of PNG file paths from VizBuilder.
            question:        Original user question (used in subject line).

        Returns:
            dict with keys:
              - "success":   bool — True if email was sent without error
              - "message":   str  — confirmation or error description
              - "recipient": str  — the address the email was sent to
        """
        missing = [
            name for name, val in {
                "SMTP_HOST":     self.smtp_host,
                "SMTP_USER":     self.smtp_user,
                "SMTP_PASSWORD": self.smtp_password,
            }.items() if not val
        ]
        if missing:
            return {
                "success":   False,
                "message":   f"Missing required environment variable(s): {', '.join(missing)}",
                "recipient": recipient_email,
            }

        try:
            msg = self._compose(recipient_email, report_text, figure_paths, question)
            self._send(msg, recipient_email)
            return {
                "success":   True,
                "message":   f"Report sent successfully to {recipient_email}",
                "recipient": recipient_email,
            }
        except Exception as exc:
            return {
                "success":   False,
                "message":   f"Failed to send email: {exc}",
                "recipient": recipient_email,
            }

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _compose(
        self,
        recipient: str,
        report_text: str,
        figure_paths: list[str],
        question: str,
    ) -> MIMEMultipart:
        """
        Build the MIME message.

        Structure:
          multipart/mixed          ← outer (for .md attachment)
          └─ multipart/related     ← for HTML + inline images
             └─ multipart/alternative
                └─ text/html      ← the email body
             └─ image/png × N     ← inline chart images (Content-ID refs)
          └─ application/octet-stream  ← .md report attachment
        """
        subject   = f"Aria Analytics Report — {question}"
        timestamp = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
        insights  = self._extract_insights(report_text)
        summary   = self._extract_summary(report_text)

        # Outer container
        outer = MIMEMultipart("mixed")
        outer["Subject"] = subject
        outer["From"]    = self.smtp_user
        outer["To"]      = recipient

        # Related container (HTML body + inline images)
        related = MIMEMultipart("related")
        outer.attach(related)

        # Assign a Content-ID to each figure
        cid_map: dict[str, str] = {}
        existing = [p for p in figure_paths if os.path.exists(p)]
        for i, path in enumerate(existing):
            cid_map[path] = f"chart{i}@aria"

        # HTML body
        html = self._build_html(question, summary, insights, cid_map, timestamp)
        alt  = MIMEMultipart("alternative")
        alt.attach(MIMEText(html, "html", "utf-8"))
        related.attach(alt)

        # Inline chart images
        for path, cid in cid_map.items():
            with open(path, "rb") as f:
                img = MIMEImage(f.read(), _subtype="png")
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline",
                           filename=Path(path).name)
            related.attach(img)

        # Markdown report as attachment
        md_bytes = report_text.encode("utf-8")
        md_part  = MIMEBase("application", "octet-stream")
        md_part.set_payload(md_bytes)
        encoders.encode_base64(md_part)
        md_part.add_header(
            "Content-Disposition",
            "attachment",
            filename=f"aria_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md",
        )
        outer.attach(md_part)

        return outer

    def _send(self, msg: MIMEMultipart, recipient: str) -> None:
        """
        Connect to SMTP and send msg.
        Uses STARTTLS for port 587, implicit SSL for port 465,
        and plain (no encryption) for everything else.
        """
        if self.smtp_port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=ctx) as server:
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, recipient, msg.as_string())
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                if self.smtp_port == 587:
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, recipient, msg.as_string())

    def _extract_summary(self, report_text: str) -> str:
        """Pull the Executive Summary paragraph from the markdown report."""
        match = re.search(
            r"## Executive Summary\s*\n(.*?)(?=\n## |\Z)",
            report_text,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        # Fallback: first non-heading paragraph
        lines = [l for l in report_text.splitlines() if l.strip() and not l.startswith("#")]
        return " ".join(lines[:3]) if lines else ""

    def _extract_insights(self, report_text: str) -> list[str]:
        """Extract up to 3 bullet points from the Key Findings section."""
        match = re.search(
            r"## Key Findings\s*\n(.*?)(?=\n## |\Z)",
            report_text,
            re.DOTALL,
        )
        if not match:
            return []
        section = match.group(1)
        # Match lines starting with - or * (markdown bullets)
        bullets = re.findall(r"^[-*]\s+(.+)", section, re.MULTILINE)
        # Strip inline markdown bold (**text**)
        cleaned = [re.sub(r"\*\*(.+?)\*\*", r"\1", b).strip() for b in bullets]
        return cleaned[:3]

    def _build_html(
        self,
        question: str,
        summary: str,
        insights: list[str],
        cid_map: dict[str, str],
        timestamp: str,
    ) -> str:
        """Render the full HTML email body."""

        insight_rows = "".join(
            f"""
            <tr>
              <td style="padding:10px 14px;border-bottom:1px solid #f0f0f0;
                         font-size:14px;line-height:1.6;color:#374151;">
                <span style="display:inline-block;background:#1a56db;color:#fff;
                             border-radius:50%;width:22px;height:22px;
                             text-align:center;line-height:22px;font-size:11px;
                             font-weight:700;margin-right:10px;">{i+1}</span>
                {insight}
              </td>
            </tr>"""
            for i, insight in enumerate(insights)
        )

        chart_blocks = "".join(
            f"""
            <div style="margin-bottom:24px;">
              <img src="cid:{cid}" alt="Chart {i+1}"
                   style="max-width:100%;border-radius:8px;
                          border:1px solid #e5e7eb;display:block;" />
            </div>"""
            for i, (path, cid) in enumerate(cid_map.items())
        )

        charts_section = f"""
            <h2 style="color:#111827;font-size:18px;margin:32px 0 16px;">
              📊 Data Visualizations
            </h2>
            {chart_blocks}
        """ if chart_blocks else ""

        findings_section = f"""
            <h2 style="color:#111827;font-size:18px;margin:32px 0 12px;">
              💡 Key Findings
            </h2>
            <table width="100%" cellpadding="0" cellspacing="0"
                   style="border:1px solid #e5e7eb;border-radius:8px;
                          border-collapse:separate;border-spacing:0;overflow:hidden;">
              {insight_rows}
            </table>
        """ if insight_rows else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,
             'Segoe UI',Helvetica,Arial,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 16px;">
    <tr><td align="center">
      <table width="640" cellpadding="0" cellspacing="0"
             style="background:#ffffff;border-radius:12px;overflow:hidden;
                    box-shadow:0 1px 8px rgba(0,0,0,0.08);">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#1a56db,#3b82f6);
                     padding:32px 40px;text-align:center;">
            <p style="margin:0;font-size:28px;font-weight:800;color:#ffffff;
                      letter-spacing:-0.5px;">📊 Aria</p>
            <p style="margin:8px 0 0;font-size:13px;color:#bfdbfe;">
              Autonomous Reasoning &amp; Insight Agent
            </p>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:36px 40px;">

            <!-- Question -->
            <div style="background:#eff6ff;border-left:4px solid #1a56db;
                        border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:28px;">
              <p style="margin:0;font-size:12px;font-weight:600;color:#1a56db;
                        text-transform:uppercase;letter-spacing:0.06em;">
                Analysis Question
              </p>
              <p style="margin:6px 0 0;font-size:15px;color:#111827;font-weight:500;">
                {question}
              </p>
            </div>

            <!-- Summary -->
            <h2 style="color:#111827;font-size:18px;margin:0 0 12px;">
              📋 Executive Summary
            </h2>
            <p style="color:#374151;font-size:14px;line-height:1.75;margin:0 0 24px;">
              {summary}
            </p>

            {findings_section}

            {charts_section}

            <!-- Attachment note -->
            <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
                        padding:16px 18px;margin-top:32px;">
              <p style="margin:0;font-size:13px;color:#6b7280;">
                📎 The full report is attached as a <strong>.md</strong> file.
                Open it in any markdown viewer or text editor.
              </p>
            </div>

          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f9fafb;border-top:1px solid #f0f0f0;
                     padding:20px 40px;text-align:center;">
            <p style="margin:0;font-size:12px;color:#9ca3af;">
              Generated by Aria on {timestamp}<br>
              Built by <strong style="color:#374151;">Kiran Reddy Konapalli</strong>
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>

</body>
</html>"""
