name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: 
  - bug
body:
  - type: markdown
    attributes:
      value: >
        Thanks for taking the time to fill out this bug report!
        
        Before submitting, please make sure you've read our [important information about reporting issues](link-to-before_reporting.md).
  - type: checkboxes
    id: terms
    attributes:
      label: Preflight Checklist
      description: Please ensure you've completed these before submitting
      options:
        - label: I have verified this is an actual bug by asking in the Discord
          required: true
        - label: I understand that GitHub issues are ONLY for verified bugs in the code
          required: true