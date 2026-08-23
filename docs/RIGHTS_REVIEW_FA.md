# راهنمای ثبت مجوز و اجازهٔ استفاده پژوهشی داده‌های مکالمه

این فرایند دو مفهوم مستقل را به‌صورت قابل ممیزی ثبت می‌کند:

1. **مجوز منبع**: مجوز صریحی مانند CC BY یا اجازهٔ دارندهٔ حقوق؛
2. **اجازهٔ پژوهشی داخلی**: تأیید استاد راهنما یا واحد مجاز دانشگاه برای آموزش
   و ارزیابی خصوصی پایان‌نامه، بدون ادعای مالکیت یا مجوز انتشار داده.

اجرای خزش، در دسترس بودن عمومی فایل و اجازهٔ پژوهشی داخلی به‌تنهایی
`license_verified=true` ایجاد نمی‌کنند. در عوض، تأیید مکتوب استاد می‌تواند
`internal_research_authorized=true` را با مبنای
`supervisor-approved-internal-research` فعال کند. در این حالت انتشار صوت و
کپشن خام همیشه ممنوع باقی می‌ماند.

تصمیم جاری و دامنهٔ دقیق آن در `docs/SUPERVISOR_DECISIONS.md` ثبت شده است.

## ایجاد فرم

```bash
.venv/bin/python -m thesis_s2s.cli create-conversation-rights-review
```

فایل
`data/processed/manifests/conversation_rights_review.csv`
یک ردیف برای هر کانال/منبع می‌سازد. مدرک هر ردیف باید تمام اپیزودهای همان
`scope_id` را پوشش دهد؛ اگر تصمیم فقط چند اپیزود را پوشش می‌دهد، دامنهٔ داده
باید محدود شود.

## فیلدهای لازم

| ستون | مقدار لازم |
|---|---|
| `approval_status` | `approved` یا `rejected` |
| `authorization_basis` | `supervisor-approved-internal-research` یا `explicit-source-license` |
| `license_name` | فقط برای مبنای مجوز صریح؛ مانند `CC-BY-4.0` |
| `internal_training_allowed` | اجازهٔ آموزش داخلی: `yes` / `no` |
| `thesis_reporting_allowed` | اجازهٔ گزارش نتایج پایان‌نامه: `yes` / `no` |
| `derived_artifacts_allowed` | اجازهٔ نگه‌داری/ارزیابی خروجی مشتق‌شده: `yes` / `no` |
| `redistribution_allowed` | فقط مجوز صریح منبع می‌تواند انتشار را مجاز کند |
| `evidence_reference` | شناسهٔ ایمیل، صورت‌جلسه، نامه، مجوز یا سند بایگانی‌شده |
| `approved_by` | نقش یا شناسهٔ فرد/واحد مجاز |
| `approval_date` | تاریخ ISO به‌شکل `YYYY-MM-DD` |
| `notes` | محدودیت‌ها یا توضیح اختیاری |

برای اجازهٔ آموزش، وضعیت باید `approved`، سه اجازهٔ اول باید `yes` و مدرک،
تأییدکننده و تاریخ باید معتبر باشند.

- با مبنای `supervisor-approved-internal-research`، مقدار
  `internal_research_authorized=true`، ولی `license_verified=false` و
  `redistribution_allowed=false` خواهد بود.
- با مبنای `explicit-source-license`، نام مجوز غیرخالی و معتبر لازم است؛ فقط
  در این حالت `license_verified=true` می‌شود و انتشار تابع همان تصمیم/مجوز است.

## پاک‌سازی متادیتای قدیمی

```bash
.venv/bin/python -m thesis_s2s.cli normalize-conversation-rights-metadata
```

این دستور اتمیک و تکرارپذیر است و وضعیت قدیمی یا ناقص را به
`pending-youtube-rights-review`، `license_verified=false`،
`internal_research_authorized=false`، `authorization_basis=pending` و
`redistribution_allowed=false` تبدیل می‌کند. هیچ مجوز یا اجازهٔ آموزشی را
استنباط نمی‌کند.

## اعمال و ممیزی

```bash
.venv/bin/python -m thesis_s2s.cli apply-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes \
  --manifest data/processed/manifests/conversation_episode_windows_approved.jsonl
```

دستور اعمال، منیفست ورودی را تغییر نمی‌دهد. تصمیم ناقص، تاریخ نامعتبر، مدرک
خالی، کانال ناشناخته، مبنای نامعتبر یا اجازهٔ ناکافی به‌صورت محافظه‌کارانه رد
می‌شود. گزارش، ساعت‌های مجاز برای آموزش و ساعت‌های دارای مجوز صریح را جداگانه
ثبت می‌کند.

ساخت جفت مکالمه و خروجی LLaMA-Omni2 فقط با مجوز صریح منبع **یا** اجازهٔ
پژوهشی داخلی کامل انجام می‌شود. هیچ‌کدام بدون مجوز صریح، انتشار دادهٔ خام را
فعال نمی‌کنند.

CSV تصمیم و مدارک اصلی ممکن است اطلاعات داخلی داشته باشند و نباید بدون اجازه
در مخزن عمومی قرار گیرند. در نسخهٔ عمومی فقط گزارش حداقلی، شناسهٔ غیرمحرمانهٔ
مدرک و آمار تجمیعی منتشر شود.
