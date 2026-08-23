# راهنمای ثبت مجوز داده‌های مکالمه

این فرایند فقط تصمیم و مدرک مجوز را به‌صورت قابل ممیزی ثبت می‌کند و جایگزین
نظر حقوقی یا اجازه مالک داده نیست. داشتن فایل در آرشیو یا امکان دانلود آن به
معنای مجاز بودن آموزش مدل نیست. دانشجو نباید بدون تأیید کتبیِ فرد یا واحد
مجاز، وضعیت را `approved` ثبت کند.

## ایجاد فرم

پس از بازبینی دستی منیفست:

```bash
.venv/bin/python -m thesis_s2s.cli create-conversation-rights-review
```

فایل
`data/processed/manifests/conversation_rights_review.csv`
یک ردیف برای هر کانال/منبع می‌سازد. مدرک معرفی‌شده در هر ردیف باید تمام
اپیزودهای همان `scope_id` را پوشش دهد؛ اگر مجوز فقط چند اپیزود را پوشش
می‌دهد، آن تصمیم برای کل کانال قابل استفاده نیست و باید دامنه داده محدود یا
مجوز دقیق‌تری دریافت شود.

## فیلدهای لازم

| ستون | مقدار لازم |
|---|---|
| `approval_status` | `approved` یا `rejected` |
| `internal_training_allowed` | اجازه آموزش داخلی: `yes` / `no` |
| `thesis_reporting_allowed` | اجازه گزارش نتایج در پایان‌نامه: `yes` / `no` |
| `derived_artifacts_allowed` | اجازه نگه‌داری/ارزیابی خروجی مشتق‌شده: `yes` / `no` |
| `redistribution_allowed` | اجازه انتشار داده؛ مستقل از استفاده داخلی |
| `evidence_reference` | شناسه نامه، ایمیل، قرارداد یا سند بایگانی‌شده |
| `approved_by` | نقش یا شناسه فرد/واحد مجاز |
| `approval_date` | تاریخ ISO به‌شکل `YYYY-MM-DD` |
| `notes` | محدودیت‌ها یا توضیح اختیاری |

برای `license_verified=true` باید وضعیت `approved` باشد، سه اجازه اول
`yes` باشند و مدرک، تأییدکننده و تاریخ معتبر وجود داشته باشند. اجازه انتشار
مستقل است؛ `redistribution_allowed=no` مانع استفاده داخلیِ تأییدشده نمی‌شود
اما انتشار صوت/کپشن را ممنوع نگه می‌دارد.

## اعمال و ممیزی

```bash
.venv/bin/python -m thesis_s2s.cli apply-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes \
  --manifest data/processed/manifests/conversation_episode_windows_approved.jsonl
```

دستور اعمال، منیفست ورودی را تغییر نمی‌دهد و فایل
`conversation_episode_windows_approved.jsonl`
و گزارش
`results/conversation_rights_report.json`
را می‌سازد. تصمیم ناقص، تاریخ نامعتبر، مدرک خالی، کانال ناشناخته یا اجازه
ناکافی به‌صورت محافظه‌کارانه رد می‌شود.

دستورهای ساخت جفت مکالمه و خروجی LLaMA-Omni2 نیز پیش از ایجاد هر خروجی، تمام
ردیف‌ها را بررسی می‌کنند و با وجود حتی یک ردیف بدون مجوز معتبر متوقف می‌شوند.

CSV مجوز و مدارک اصلی ممکن است اطلاعات داخلی داشته باشند و نباید بدون اجازه
در مخزن عمومی قرار گیرند. در نسخه عمومی فقط گزارش حداقلی و غیرمحرمانه یا
شناسه مدرک منتشر شود.
