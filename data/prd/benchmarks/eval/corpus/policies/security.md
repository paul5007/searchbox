# Security Policy

All employee access to internal systems is granted through single sign-on (SSO) with mandatory
multi-factor authentication (MFA). Production system access additionally requires connection
through the corporate VPN. Secrets and API keys are stored in HashiCorp Vault, never in source
control. Access reviews are conducted quarterly by the security team.
