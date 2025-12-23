FROM python:3.11-slim

# 1) 安装 Node.js 18 + 基础构建工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
 && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# 2) 安装 uv（MiroFish 后端依赖它）
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

# 3) 安装所有依赖（根目录 + 前端 + 后端）
# 仓库推荐的一键安装命令
RUN npm run setup:all

# 4) 暴露端口：前端 3000，后端 5001
EXPOSE 3000 5001

# 5) 启动（和仓库一致）
CMD ["npm", "run", "dev"]
