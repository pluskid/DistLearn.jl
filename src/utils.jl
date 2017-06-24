function try_fetch(future :: Future)
  ret = fetch(future)
  if ret isa RemoteException
    throw(ret)
  end
  return ret
end
