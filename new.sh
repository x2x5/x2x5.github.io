#!/bin/bash

# 这是一个脚本，用于创建带有 YYYY-MM-DD 前缀的文章目录

# 检查用户是否提供了文章标题参数
if [ -z "$1" ]; then
  echo "❌ 错误: 请提供一个文章标题 (slug)."
  echo "   用法: ./hugo-post.sh my-new-post"
  exit 1
fi

# 1. 获取当前日期，格式为 YYYY-MM-DD
current_date=$(date +%Y-%m-%d)

# 2. 从第一个参数获取文章标题
title_slug="$1"

# 3. 拼接成最终的目录名
full_directory_name="${current_date}-${title_slug}"

mkdir -p "content/posts/${full_directory_name}"


# 4. 调用 hugo new 命令，注意末尾的斜杠以创建 Page Bundle
hugo new "posts/${full_directory_name}/index.md"

# 5. 打印成功信息
echo "✅ 成功创建文章: content/posts/${full_directory_name}/"